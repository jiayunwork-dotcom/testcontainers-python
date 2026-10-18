import io
import itertools
import logging
import re
import time
from datetime import timedelta
from email.message import Message
from types import SimpleNamespace
from typing import Any, Optional, cast
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import pytest

from testcontainers.compose import ComposeContainer
from testcontainers.core.exceptions import ContainerExitedException
from testcontainers.core.inspect import ContainerInspectInfo, ContainerState
from testcontainers.core.wait_strategies import (
    CompositeWaitStrategy,
    ContainerStatusWaitStrategy,
    ExecWaitStrategy,
    FileExistsWaitStrategy,
    HealthcheckWaitStrategy,
    HttpWaitStrategy,
    LogMessageWaitStrategy,
    PortWaitStrategy,
    WaitStrategy,
    WaitStrategyTarget,
)
from testcontainers.core.waiting_utils import get_container_exit_state, raise_if_container_exited


class ConcreteWaitStrategy(WaitStrategy):
    """Concrete implementation for testing abstract base class."""

    def wait_until_ready(self, container: "WaitStrategyTarget") -> None:
        # Simple implementation that just waits a bit
        time.sleep(0.1)


class TestWaitStrategy:
    """Test the base WaitStrategy class."""

    def test_wait_strategy_initialization(self):
        strategy = ConcreteWaitStrategy()
        assert strategy._startup_timeout > 0
        assert strategy._poll_interval > 0

    @pytest.mark.parametrize(
        "timeout_value,expected_seconds",
        [
            (30, 30),
            (timedelta(seconds=45), 45),
            (60, 60),
            (timedelta(minutes=2), 120),
        ],
        ids=[
            "timeout_int_30_seconds",
            "timeout_timedelta_45_seconds",
            "timeout_int_60_seconds",
            "timeout_timedelta_2_minutes",
        ],
    )
    def test_with_startup_timeout(self, timeout_value, expected_seconds):
        strategy = ConcreteWaitStrategy()
        result = strategy.with_startup_timeout(timeout_value)
        assert result is strategy
        assert strategy._startup_timeout == expected_seconds

    @pytest.mark.parametrize(
        "interval_value,expected_seconds",
        [
            (2.5, 2.5),
            (timedelta(seconds=3), 3.0),
            (0.1, 0.1),
            (timedelta(milliseconds=500), 0.5),
        ],
        ids=[
            "interval_float_2_5_seconds",
            "interval_timedelta_3_seconds",
            "interval_float_0_1_seconds",
            "interval_timedelta_500_milliseconds",
        ],
    )
    def test_with_poll_interval(self, interval_value, expected_seconds):
        strategy = ConcreteWaitStrategy()
        result = strategy.with_poll_interval(interval_value)
        assert result is strategy
        assert strategy._poll_interval == expected_seconds

    def test_abstract_method(self):
        # Test that abstract base class cannot be instantiated
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            WaitStrategy()  # type: ignore[abstract]


class TestLogMessageWaitStrategy:
    """Test the LogMessageWaitStrategy class."""

    @pytest.mark.parametrize(
        "message,times,predicate_streams_and",
        [
            ("test message", 1, False),
            (re.compile(r"test\d+"), 1, False),
            ("test", 3, False),
            ("test", 1, True),
            ("ready", 2, True),
        ],
        ids=[
            "simple_string_message",
            "regex_pattern_message",
            "message_with_times_3",
            "message_with_predicate_streams_and_true",
            "ready_message_with_times_and_predicate",
        ],
    )
    def test_log_message_wait_strategy_initialization(self, message, times, predicate_streams_and):
        strategy = LogMessageWaitStrategy(message, times=times, predicate_streams_and=predicate_streams_and)

        if isinstance(message, str):
            assert strategy._message.pattern == message
        else:
            assert strategy._message is message

        assert strategy._times == times
        assert strategy._predicate_streams_and is predicate_streams_and

    @pytest.mark.parametrize(
        "container_logs,expected_message,should_succeed",
        [
            ((b"test message", b""), "test message", True),
            ((b"", b"test message"), "test message", True),
            ((b"no match", b""), "test message", False),
            ((b"test123", b""), re.compile(r"test\d+"), True),
            ((b"test", b""), re.compile(r"test\d+"), False),
        ],
        ids=[
            "stdout_contains_message_success",
            "stderr_contains_message_success",
            "no_message_match_failure",
            "regex_pattern_match_success",
            "regex_pattern_no_match_failure",
        ],
    )
    @patch("time.time")
    @patch("time.sleep")
    def test_wait_until_ready(self, mock_sleep, mock_time, container_logs, expected_message, should_succeed):
        strategy = LogMessageWaitStrategy(expected_message)
        mock_container = Mock()
        mock_container.get_logs.return_value = container_logs
        # Mock the wrapped container to simulate a running container
        mock_wrapped = Mock()
        mock_wrapped.status = "running"
        mock_wrapped.reload.return_value = None
        mock_container.get_wrapped_container.return_value = mock_wrapped
        # Configure time mock to simulate timeout for failure cases
        if should_succeed:
            mock_time.side_effect = [0, 1]
        else:
            mock_time.side_effect = itertools.count(start=0, step=1)
        if should_succeed:
            strategy.wait_until_ready(mock_container)
            mock_container.get_logs.assert_called_once()
        else:
            with pytest.raises(TimeoutError):
                strategy.wait_until_ready(mock_container)


class TestHttpWaitStrategy:
    """Test the HttpWaitStrategy class."""

    @pytest.mark.parametrize(
        "port,path,expected_port,expected_path",
        [
            (8080, "/health", 8080, "/health"),
            (80, None, 80, "/"),
            (443, "/api/status", 443, "/api/status"),
            (3000, "", 3000, "/"),
        ],
        ids=[
            "port_8080_health_path",
            "port_80_default_path",
            "port_443_api_status_path",
            "port_3000_empty_path",
        ],
    )
    def test_http_wait_strategy_initialization(self, port, path, expected_port, expected_path):
        strategy = HttpWaitStrategy(port, path)
        assert strategy._port == expected_port
        assert strategy._path == expected_path
        assert strategy._status_codes == {200}
        assert strategy._method == "GET"

    @pytest.mark.parametrize(
        "status_codes",
        [
            [404],
            [200, 201],
            [500, 502, 503],
            [200, 404, 500],
        ],
        ids=[
            "single_status_code_404",
            "multiple_status_codes_200_201",
            "error_status_codes_500_502_503",
            "mixed_status_codes_200_404_500",
        ],
    )
    def test_for_status_code(self, status_codes):
        strategy = HttpWaitStrategy(8080)

        for code in status_codes:
            result = strategy.for_status_code(code)
            assert result is strategy
            assert code in strategy._status_codes

        # Default 200 should still be there
        assert 200 in strategy._status_codes

    @pytest.mark.parametrize(
        "predicate_description,status_code_predicate,response_predicate",
        [
            ("status_code_range", lambda code: 200 <= code < 300, None),
            ("status_code_equals_200", lambda code: code == 200, None),
            ("response_contains_ready", None, lambda body: "ready" in body),
            ("response_json_valid", None, lambda body: "status" in body),
            ("both_predicates", lambda code: code >= 200, lambda body: len(body) > 0),
        ],
        ids=[
            "status_code_range_200_to_300",
            "status_code_equals_200",
            "response_contains_ready",
            "response_json_valid",
            "both_status_and_response_predicates",
        ],
    )
    def test_predicates(self, predicate_description, status_code_predicate, response_predicate):
        strategy = HttpWaitStrategy(8080)

        if status_code_predicate:
            result = strategy.for_status_code_matching(status_code_predicate)
            assert result is strategy
            assert strategy._status_code_predicate is status_code_predicate

        if response_predicate:
            result = strategy.for_response_predicate(response_predicate)
            assert result is strategy
            assert strategy._response_predicate is response_predicate

    @pytest.mark.parametrize(
        "tls_config,expected_tls,expected_insecure",
        [
            ({"insecure": True}, True, True),
            ({"insecure": False}, True, False),
            ({}, True, False),
        ],
        ids=[
            "tls_insecure_true",
            "tls_insecure_false",
            "tls_default_insecure_false",
        ],
    )
    def test_using_tls(self, tls_config, expected_tls, expected_insecure):
        strategy = HttpWaitStrategy(8080)
        result = strategy.using_tls(**tls_config)
        assert result is strategy
        assert strategy._tls is expected_tls
        assert strategy._insecure_tls is expected_insecure

    @pytest.mark.parametrize(
        "headers",
        [
            {"Authorization": "Bearer token"},
            {"Content-Type": "application/json"},
            {"User-Agent": "test", "Accept": "text/html"},
        ],
        ids=[
            "single_header_authorization",
            "single_header_content_type",
            "multiple_headers_user_agent_accept",
        ],
    )
    def test_with_header(self, headers):
        strategy = HttpWaitStrategy(8080)

        for key, value in headers.items():
            result = strategy.with_header(key, value)
            assert result is strategy
            assert strategy._headers[key] == value

    @pytest.mark.parametrize(
        "credentials",
        [
            ("user", "pass"),
            ("admin", "secret123"),
            ("test", ""),
        ],
        ids=[
            "basic_credentials_user_pass",
            "basic_credentials_admin_secret",
            "basic_credentials_test_empty",
        ],
    )
    def test_with_basic_credentials(self, credentials):
        strategy = HttpWaitStrategy(8080)
        result = strategy.with_basic_credentials(*credentials)
        assert result is strategy
        assert strategy._basic_auth == credentials

    @pytest.mark.parametrize(
        "method",
        [
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "HEAD",
        ],
        ids=[
            "method_get",
            "method_post",
            "method_put",
            "method_delete",
            "method_head",
        ],
    )
    def test_with_method(self, method):
        strategy = HttpWaitStrategy(8080)
        result = strategy.with_method(method)
        assert result is strategy
        assert strategy._method == method

    @pytest.mark.parametrize(
        "body",
        [
            '{"key": "value"}',
            '{"status": "ready"}',
            "data=test&format=json",
            "",
        ],
        ids=[
            "json_body_key_value",
            "json_body_status_ready",
            "form_data_body",
            "empty_body",
        ],
    )
    def test_with_body(self, body):
        strategy = HttpWaitStrategy(8080)
        result = strategy.with_body(body)
        assert result is strategy
        assert strategy._body == body

    @pytest.mark.parametrize(
        "url,expected_port,expected_path,expected_tls",
        [
            ("https://localhost:8080/api/health", 8080, "/api/health", True),
            ("http://localhost:3000", 3000, "/", False),
            ("https://example.com", 443, "/", True),
            ("http://localhost:80/", 80, "/", False),
        ],
        ids=[
            "https_localhost_8080_api_health",
            "http_localhost_3000_default_path",
            "https_example_com_default_port",
            "http_localhost_80_root_path",
        ],
    )
    def test_from_url(self, url, expected_port, expected_path, expected_tls):
        strategy = HttpWaitStrategy.from_url(url)
        assert strategy._port == expected_port
        assert strategy._path == expected_path
        assert strategy._tls is expected_tls

    @pytest.mark.parametrize(
        "status_codes,expected_result",
        [
            ({503}, True),
            (set(), False),
        ],
        ids=[
            "accepted_error_status_code",
            "unaccepted_error_status_code",
        ],
    )
    @patch("testcontainers.core.wait_strategies.urlopen")
    def test_try_http_request_closes_http_error(self, mock_urlopen, status_codes, expected_result):
        """HTTPError holds the response's file object and must be closed (issue #1115)."""
        fp = io.BytesIO(b"error body")
        mock_urlopen.side_effect = HTTPError("http://localhost:8080/", 503, "Service Unavailable", Message(), fp)
        strategy = HttpWaitStrategy(8080)
        for code in status_codes:
            strategy.for_status_code(code)

        result = strategy._try_http_request("http://localhost:8080/", {}, None)

        assert result is expected_result
        assert fp.closed


class TestHealthcheckWaitStrategy:
    """Test the HealthcheckWaitStrategy class."""

    def test_healthcheck_wait_strategy_initialization(self):
        strategy = HealthcheckWaitStrategy()
        # Should inherit from WaitStrategy
        assert hasattr(strategy, "_startup_timeout")
        assert hasattr(strategy, "_poll_interval")

    @pytest.mark.parametrize(
        "health_status,health_log,should_succeed,expected_error",
        [
            ("healthy", None, True, None),
            (
                "unhealthy",
                [{"ExitCode": 1, "Output": "Health check failed"}],
                False,
                "Container health check reported unhealthy",
            ),
            ("starting", None, False, "Container health check did not report healthy within 120.* seconds"),
            (None, None, False, "No health check configured"),
        ],
        ids=[
            "healthy_status_success",
            "unhealthy_status_failure",
            "starting_status_failure",
            "no_healthcheck_failure",
        ],
    )
    @patch("time.time")
    @patch("time.sleep")
    def test_wait_until_ready(self, mock_sleep, mock_time, health_status, health_log, should_succeed, expected_error):
        strategy = HealthcheckWaitStrategy()
        mock_container = Mock()

        # Mock the wrapped container
        mock_wrapped = Mock()
        mock_wrapped.status = "running"
        mock_wrapped.reload.return_value = None

        # Mock health check data
        health_data = {}
        if health_status:
            health_data = {"Status": health_status}
            if health_log:
                health_data["Log"] = health_log

        mock_wrapped.attrs = {"State": {"Health": health_data}}

        mock_container.get_wrapped_container.return_value = mock_wrapped

        # Configure time mock based on expected behavior
        if should_succeed:
            mock_time.side_effect = [0, 1]
        else:
            # For failure cases, we need more time values to handle the loop
            mock_time.side_effect = itertools.count(start=0, step=1)  # Provide enough values for the loop

        if should_succeed:
            strategy.wait_until_ready(mock_container)
        else:
            with pytest.raises((RuntimeError, TimeoutError), match=expected_error):
                strategy.wait_until_ready(mock_container)


class TestPortWaitStrategy:
    """Test the PortWaitStrategy class."""

    @pytest.mark.parametrize(
        "port",
        [
            8080,
            80,
            443,
            22,
            3306,
        ],
        ids=[
            "port_8080",
            "port_80",
            "port_443",
            "port_22",
            "port_3306",
        ],
    )
    def test_port_wait_strategy_initialization(self, port):
        strategy = PortWaitStrategy(port)
        assert strategy._port == port

    @pytest.mark.parametrize(
        "connection_success,expected_behavior",
        [
            (True, "success"),
            (False, "timeout"),
        ],
        ids=[
            "socket_connection_success",
            "socket_connection_timeout",
        ],
    )
    @patch("socket.socket")
    @patch("time.time")
    @patch("time.sleep")
    def test_wait_until_ready(self, mock_sleep, mock_time, mock_socket, connection_success, expected_behavior):
        strategy = PortWaitStrategy(8080).with_startup_timeout(1)
        mock_container = Mock()
        mock_container.get_container_host_ip.return_value = "localhost"
        mock_container.get_exposed_port.return_value = 8080

        # Mock socket connection
        mock_socket_instance = Mock()
        if connection_success:
            mock_socket.return_value.__enter__.return_value = mock_socket_instance
            mock_time.side_effect = [0, 1]
        else:
            mock_socket_instance.connect.side_effect = ConnectionRefusedError()
            mock_socket.return_value.__enter__.return_value = mock_socket_instance
            mock_time.side_effect = [0, 2]  # Exceed timeout

        if expected_behavior == "success":
            strategy.wait_until_ready(mock_container)
            mock_socket_instance.connect.assert_called_once_with(("localhost", 8080))
        else:
            with pytest.raises(TimeoutError, match=r"Port 8080 not available within 1.0 seconds"):
                strategy.wait_until_ready(mock_container)


class TestFileExistsWaitStrategy:
    """Test the FileExistsWaitStrategy class."""

    @pytest.mark.parametrize(
        "file_path",
        [
            "/tmp/test.txt",
            "/var/log/app.log",
            "/opt/app/config.yaml",
            "relative/path/file.conf",
        ],
        ids=[
            "tmp_file_path",
            "var_log_file_path",
            "opt_config_file_path",
            "relative_file_path",
        ],
    )
    def test_file_exists_wait_strategy_initialization(self, file_path):
        strategy = FileExistsWaitStrategy(file_path)
        # _file_path is stored as a Path object
        assert str(strategy._file_path) == file_path
        # Should inherit from WaitStrategy
        assert hasattr(strategy, "_startup_timeout")
        assert hasattr(strategy, "_poll_interval")

    @pytest.mark.parametrize(
        "file_exists,expected_behavior",
        [
            (True, "success"),
            (False, "timeout"),
        ],
        ids=[
            "file_exists_success",
            "file_not_exists_timeout",
        ],
    )
    @patch("pathlib.Path.is_file")
    @patch("time.time")
    @patch("time.sleep")
    def test_wait_until_ready(self, mock_sleep, mock_time, mock_is_file, file_exists, expected_behavior, caplog):
        strategy = FileExistsWaitStrategy("/tmp/test.txt").with_startup_timeout(1)
        mock_container = Mock()

        # Configure mocks based on expected behavior
        if file_exists:
            mock_is_file.return_value = True
            # Need multiple time values for debug logging
            mock_time.side_effect = [0, 0.1, 0.2]  # Start time, check time, logging time
        else:
            mock_is_file.return_value = False
            # Need more time values for the loop and logging calls
            mock_time.side_effect = itertools.count(start=0, step=0.6)  # Exceed timeout after a few iterations

        if expected_behavior == "success":
            strategy.wait_until_ready(mock_container)
            mock_is_file.assert_called()
        else:
            with pytest.raises(TimeoutError, match=r"File.*did not exist within.*seconds"):
                with caplog.at_level(logging.CRITICAL, logger="testcontainers.core.wait_strategies"):
                    strategy.wait_until_ready(mock_container)


class TestCompositeWaitStrategy:
    """Test the CompositeWaitStrategy class."""

    def test_composite_wait_strategy_initialization_single_strategy(self):
        """Test initialization with a single strategy."""
        log_strategy = LogMessageWaitStrategy("ready")
        composite = CompositeWaitStrategy(log_strategy)
        assert composite._strategies == [log_strategy]

    def test_composite_wait_strategy_initialization_multiple_strategies(self):
        """Test initialization with multiple strategies."""
        log_strategy = LogMessageWaitStrategy("ready")
        port_strategy = PortWaitStrategy(8080)
        file_strategy = FileExistsWaitStrategy("/tmp/test.txt")

        composite = CompositeWaitStrategy(log_strategy, port_strategy, file_strategy)
        assert composite._strategies == [log_strategy, port_strategy, file_strategy]

    def test_composite_wait_strategy_initialization_empty(self):
        """Test that empty initialization works (creates empty list)."""
        composite = CompositeWaitStrategy()
        assert composite._strategies == []

    def test_with_startup_timeout_propagates_to_child_strategies(self):
        """Test that timeout setting propagates to child strategies."""
        log_strategy = LogMessageWaitStrategy("ready")
        composite = CompositeWaitStrategy(log_strategy)
        result = composite.with_startup_timeout(30)
        assert result is composite
        assert composite._startup_timeout == 30
        # Should also propagate to child strategies
        assert log_strategy._startup_timeout == 30

    def test_with_poll_interval_propagates_to_child_strategies(self):
        """Test that poll interval setting propagates to child strategies."""
        port_strategy = PortWaitStrategy(8080)
        composite = CompositeWaitStrategy(port_strategy)
        result = composite.with_poll_interval(2.0)
        assert result is composite
        assert composite._poll_interval == 2.0
        # Should also propagate to child strategies
        assert port_strategy._poll_interval == 2.0

    def test_wait_until_ready_all_strategies_succeed(self):
        """Test that all strategies are executed when they all succeed."""
        # Create mock strategies
        strategy1 = Mock()
        strategy2 = Mock()
        strategy3 = Mock()

        composite = CompositeWaitStrategy(strategy1, strategy2, strategy3)
        mock_container = Mock()

        # All strategies should succeed
        strategy1.wait_until_ready.return_value = None
        strategy2.wait_until_ready.return_value = None
        strategy3.wait_until_ready.return_value = None

        composite.wait_until_ready(mock_container)

        # Verify all strategies were called in order
        strategy1.wait_until_ready.assert_called_once_with(mock_container)
        strategy2.wait_until_ready.assert_called_once_with(mock_container)
        strategy3.wait_until_ready.assert_called_once_with(mock_container)

    def test_wait_until_ready_first_strategy_fails(self, caplog):
        """Test that execution stops when first strategy fails."""
        strategy1 = Mock()
        strategy2 = Mock()
        strategy3 = Mock()

        composite = CompositeWaitStrategy(strategy1, strategy2, strategy3)
        mock_container = Mock()

        # First strategy fails
        strategy1.wait_until_ready.side_effect = TimeoutError("First strategy failed")

        with pytest.raises(TimeoutError, match="First strategy failed"):
            with caplog.at_level(logging.CRITICAL, logger="testcontainers.core.wait_strategies"):
                composite.wait_until_ready(mock_container)

        # Only first strategy should be called
        strategy1.wait_until_ready.assert_called_once_with(mock_container)
        strategy2.wait_until_ready.assert_not_called()
        strategy3.wait_until_ready.assert_not_called()

    def test_wait_until_ready_middle_strategy_fails(self):
        """Test that execution stops when middle strategy fails."""
        strategy1 = Mock()
        strategy2 = Mock()
        strategy3 = Mock()

        composite = CompositeWaitStrategy(strategy1, strategy2, strategy3)
        mock_container = Mock()

        # First succeeds, second fails
        strategy1.wait_until_ready.return_value = None
        strategy2.wait_until_ready.side_effect = RuntimeError("Second strategy failed")

        with pytest.raises(RuntimeError, match="Second strategy failed"):
            composite.wait_until_ready(mock_container)

        # First two strategies should be called
        strategy1.wait_until_ready.assert_called_once_with(mock_container)
        strategy2.wait_until_ready.assert_called_once_with(mock_container)
        strategy3.wait_until_ready.assert_not_called()

    @pytest.mark.parametrize(
        "strategy_types,expected_count",
        [
            (["log"], 1),
            (["log", "port"], 2),
            (["log", "port", "file"], 3),
            (["file", "log", "port", "file"], 4),
        ],
        ids=[
            "single_log_strategy",
            "log_and_port_strategies",
            "three_different_strategies",
            "four_strategies_with_duplicate_type",
        ],
    )
    def test_composite_strategy_count(self, strategy_types, expected_count):
        """Test that composite strategy handles different numbers of strategies."""
        strategies: list[WaitStrategy] = []
        for strategy_type in strategy_types:
            if strategy_type == "log":
                strategies.append(LogMessageWaitStrategy("ready"))
            elif strategy_type == "port":
                strategies.append(PortWaitStrategy(8080))
            elif strategy_type == "file":
                strategies.append(FileExistsWaitStrategy("/tmp/test.txt"))

        composite = CompositeWaitStrategy(*strategies)
        assert len(composite._strategies) == expected_count
        assert composite._strategies == strategies


def _container_state(
    status: Optional[str] = "running", exit_code: Optional[int] = None, oom_killed: bool = False
) -> dict[str, Any]:
    return {"Status": status, "ExitCode": exit_code, "OOMKilled": oom_killed}


class FakeWrappedContainer:
    """Minimal docker-py ``Container`` stand-in driven by a sequence of states."""

    def __init__(self, state_sequence: list[dict[str, Any]]) -> None:
        self._states = list(state_sequence)
        self._index = 0
        self.attrs: dict[str, Any] = {"State": self._states[0]}

    def reload(self) -> None:
        self._index = min(self._index + 1, len(self._states) - 1)
        self.attrs = {"State": self._states[self._index]}


class FakeWaitTarget:
    """Minimal :class:`WaitStrategyTarget` backed by a :class:`FakeWrappedContainer`."""

    def __init__(self, wrapped: FakeWrappedContainer, exec_result: Any = None) -> None:
        self._wrapped = wrapped
        self._exec_result = exec_result
        self.exec_calls = 0

    def get_wrapped_container(self) -> FakeWrappedContainer:
        return self._wrapped

    def reload(self) -> None:
        self._wrapped.reload()

    def get_container_host_ip(self) -> str:
        return "127.0.0.1"

    def get_exposed_port(self, port: int) -> int:
        return 9000

    def get_logs(self) -> tuple[bytes, bytes]:
        return b"", b""

    def exec(self, command: Any) -> Any:
        self.exec_calls += 1
        return self._exec_result

    @property
    def status(self) -> str:
        status = self._wrapped.attrs["State"]["Status"]
        return str(status)


class FakeComposeSnapshot:
    """Minimal ``ComposeContainer``-like snapshot (plain attributes, refreshed by reload)."""

    def __init__(self, state_sequence: list[dict[str, Any]]) -> None:
        self._states = list(state_sequence)
        self._index = 0
        state = self._states[0]
        self.State: Optional[str] = state["Status"]
        self.ExitCode: Optional[int] = state["ExitCode"]
        self.OOMKilled: Optional[bool] = state["OOMKilled"]

    def reload(self) -> None:
        self._index = min(self._index + 1, len(self._states) - 1)
        state = self._states[self._index]
        self.State = state["Status"]
        self.ExitCode = state["ExitCode"]
        self.OOMKilled = state["OOMKilled"]


class FakeComposeTarget:
    def __init__(self, snapshot: FakeComposeSnapshot) -> None:
        self._snapshot = snapshot

    def get_wrapped_container(self) -> FakeComposeSnapshot:
        return self._snapshot

    def reload(self) -> None:
        self._snapshot.reload()

    def get_container_host_ip(self) -> str:
        return "127.0.0.1"

    def get_exposed_port(self, port: int) -> int:
        return 9000

    def get_logs(self) -> tuple[bytes, bytes]:
        return b"", b""

    @property
    def status(self) -> str:
        return str(self._snapshot.State)


class TestContainerExitDetection:
    """Shared exit-state detection used by the HTTP/port/exec/file strategies."""

    @pytest.mark.parametrize(
        "state,expected_code,expected_oom",
        [
            (_container_state("exited", 1), 1, False),
            (_container_state("exited", 0), 0, False),
            (_container_state("exited", 137, True), 137, True),
            (_container_state("dead", 139), 139, False),
        ],
        ids=["exited_code_1", "exited_code_0", "exited_oom_137", "dead_code_139"],
    )
    def test_detects_terminal_states_with_exit_code(self, state, expected_code, expected_oom):
        target = FakeWaitTarget(FakeWrappedContainer([state]))
        exit_state = get_container_exit_state(target)
        assert exit_state is not None
        assert exit_state.status == state["Status"]
        assert exit_state.exit_code == expected_code
        assert exit_state.oom_killed is expected_oom

    @pytest.mark.parametrize(
        "status",
        ["running", "created", "paused", "restarting", "removing", "unknown", None],
        ids=["running", "created", "paused", "restarting", "removing", "unknown", "missing"],
    )
    def test_non_terminal_and_unknown_states_are_not_exits(self, status):
        target = FakeWaitTarget(FakeWrappedContainer([_container_state(status)]))
        assert get_container_exit_state(target) is None

    def test_unreadable_state_is_not_an_exit(self):
        target = Mock()
        target.get_wrapped_container.side_effect = RuntimeError("docker daemon unavailable")
        assert get_container_exit_state(target) is None

    def test_reload_failure_is_not_an_exit(self, monkeypatch: pytest.MonkeyPatch):
        wrapped = FakeWrappedContainer([_container_state("exited", 1)])
        target = FakeWaitTarget(wrapped)
        monkeypatch.setattr(target, "reload", Mock(side_effect=RuntimeError("docker daemon unavailable")))
        assert get_container_exit_state(target) is None

    def test_compose_snapshot_is_supported(self):
        snapshot = FakeComposeSnapshot(
            [
                _container_state("running"),
                _container_state("running"),
                _container_state("exited", 137, True),
            ]
        )
        target = FakeComposeTarget(snapshot)
        assert get_container_exit_state(target) is None
        exit_state = get_container_exit_state(target)
        assert exit_state is not None
        assert exit_state.exit_code == 137
        assert exit_state.oom_killed is True

    def test_raise_message_contains_exit_code(self):
        target = FakeWaitTarget(FakeWrappedContainer([_container_state("exited", 1)]))
        with pytest.raises(ContainerExitedException, match=r"exit code 1"):
            raise_if_container_exited(target)

    def test_raise_message_contains_exit_code_zero(self):
        target = FakeWaitTarget(FakeWrappedContainer([_container_state("exited", 0)]))
        with pytest.raises(ContainerExitedException, match=r"exit code 0"):
            raise_if_container_exited(target)

    def test_raise_message_mentions_oom(self):
        target = FakeWaitTarget(FakeWrappedContainer([_container_state("exited", 137, True)]))
        with pytest.raises(ContainerExitedException, match=r"exit code 137 \(OOM killed\)"):
            raise_if_container_exited(target)

    def test_unknown_exit_code_is_reported_as_unknown(self):
        target = FakeWaitTarget(FakeWrappedContainer([{"Status": "exited", "OOMKilled": False}]))
        with pytest.raises(ContainerExitedException, match=r"exit code unknown"):
            raise_if_container_exited(target)


class TestWaitStrategiesFailFastOnExit:
    """HTTP/port/exec/file strategies fail on the next poll when the container exits."""

    @pytest.mark.parametrize(
        "exit_code,oom",
        [(1, False), (0, False), (137, True)],
        ids=["crash_code_1", "clean_code_0", "oom_code_137"],
    )
    @patch("testcontainers.core.wait_strategies.urlopen")
    def test_http_strategy_fails_on_next_poll_when_container_exits(self, mock_urlopen, exit_code, oom):
        mock_urlopen.side_effect = URLError(ConnectionRefusedError())
        target = FakeWaitTarget(
            FakeWrappedContainer(
                [_container_state("running"), _container_state("running"), _container_state("exited", exit_code, oom)]
            )
        )
        strategy = HttpWaitStrategy(8080).with_startup_timeout(30).with_poll_interval(0.01)

        start = time.monotonic()
        with pytest.raises(ContainerExitedException, match=rf"exit code {exit_code}"):
            strategy.wait_until_ready(target)
        assert time.monotonic() - start < 5
        # Exactly one readiness probe ran before the exit was observed on the next poll.
        assert mock_urlopen.call_count == 1

    @patch("testcontainers.core.wait_strategies.urlopen")
    def test_http_strategy_fails_immediately_when_container_already_exited(self, mock_urlopen):
        mock_urlopen.side_effect = URLError(ConnectionRefusedError())
        target = FakeWaitTarget(FakeWrappedContainer([_container_state("exited", 2)]))
        strategy = HttpWaitStrategy(8080).with_startup_timeout(30).with_poll_interval(0.01)

        with pytest.raises(ContainerExitedException, match=r"exit code 2"):
            strategy.wait_until_ready(target)
        assert mock_urlopen.call_count == 0

    @patch("testcontainers.core.wait_strategies.urlopen")
    def test_http_strategy_timeout_text_unchanged_while_running(self, mock_urlopen):
        mock_urlopen.side_effect = URLError(ConnectionRefusedError())
        target = FakeWaitTarget(FakeWrappedContainer([_container_state("running")]))
        strategy = HttpWaitStrategy(8080).with_startup_timeout(1).with_poll_interval(0.01)

        with pytest.raises(TimeoutError, match=r"HTTP endpoint not ready within 1.0 seconds"):
            strategy.wait_until_ready(target)

    @patch("socket.socket")
    def test_port_strategy_fails_on_next_poll_when_container_exits(self, mock_socket):
        mock_socket.return_value.__enter__.return_value.connect.side_effect = ConnectionRefusedError()
        target = FakeWaitTarget(
            FakeWrappedContainer(
                [_container_state("running"), _container_state("running"), _container_state("exited", 137, True)]
            )
        )
        strategy = PortWaitStrategy(9000).with_startup_timeout(30).with_poll_interval(0.01)

        start = time.monotonic()
        with pytest.raises(ContainerExitedException, match=r"exit code 137 \(OOM killed\)"):
            strategy.wait_until_ready(target)
        assert time.monotonic() - start < 5
        assert mock_socket.return_value.__enter__.return_value.connect.call_count == 1

    @patch("socket.socket")
    def test_port_strategy_timeout_text_unchanged_while_running(self, mock_socket):
        mock_socket.return_value.__enter__.return_value.connect.side_effect = ConnectionRefusedError()
        target = FakeWaitTarget(FakeWrappedContainer([_container_state("running")]))
        strategy = PortWaitStrategy(9000).with_startup_timeout(1).with_poll_interval(0.01)

        with pytest.raises(TimeoutError, match=r"Port 9000 not available within 1.0 seconds"):
            strategy.wait_until_ready(target)

    def test_file_strategy_fails_on_next_poll_when_container_exits(self, monkeypatch):
        monkeypatch.setattr("pathlib.Path.is_file", lambda self: False)
        target = FakeWaitTarget(
            FakeWrappedContainer(
                [_container_state("running"), _container_state("running"), _container_state("exited", 0)]
            )
        )
        strategy = FileExistsWaitStrategy("/tmp/marker").with_startup_timeout(30).with_poll_interval(0.01)

        start = time.monotonic()
        with pytest.raises(ContainerExitedException, match=r"exit code 0"):
            strategy.wait_until_ready(target)
        assert time.monotonic() - start < 5

    def test_file_strategy_fails_immediately_when_container_already_exited(self, monkeypatch):
        checked = Mock(return_value=False)
        monkeypatch.setattr("pathlib.Path.is_file", checked)
        target = FakeWaitTarget(FakeWrappedContainer([_container_state("exited", 0)]))
        strategy = FileExistsWaitStrategy("/tmp/marker").with_startup_timeout(30).with_poll_interval(0.01)

        with pytest.raises(ContainerExitedException, match=r"exit code 0"):
            strategy.wait_until_ready(target)
        assert checked.call_count == 0

    def test_file_strategy_timeout_text_unchanged_while_running(self, monkeypatch, caplog):
        monkeypatch.setattr("pathlib.Path.is_file", lambda self: False)
        target = FakeWaitTarget(FakeWrappedContainer([_container_state("running")]))
        strategy = FileExistsWaitStrategy("/tmp/marker").with_startup_timeout(1).with_poll_interval(0.01)

        with pytest.raises(TimeoutError, match=r"File /tmp/marker did not exist within 1.000 seconds"):
            with caplog.at_level(logging.CRITICAL, logger="testcontainers.core.wait_strategies"):
                strategy.wait_until_ready(target)

    def test_exec_strategy_fails_on_next_poll_when_container_exits(self):
        target = FakeWaitTarget(
            FakeWrappedContainer(
                [_container_state("running"), _container_state("running"), _container_state("exited", 7)]
            ),
            exec_result=SimpleNamespace(exit_code=1, output=b"not ready"),
        )
        strategy = ExecWaitStrategy(["check-ready"]).with_startup_timeout(30).with_poll_interval(0.01)

        start = time.monotonic()
        with pytest.raises(ContainerExitedException, match=r"exit code 7"):
            strategy.wait_until_ready(target)
        assert time.monotonic() - start < 5
        assert target.exec_calls == 1

    def test_exec_strategy_fails_immediately_when_container_already_exited(self):
        target = FakeWaitTarget(
            FakeWrappedContainer([_container_state("exited", 2)]),
            exec_result=SimpleNamespace(exit_code=0, output=b""),
        )
        strategy = ExecWaitStrategy(["check-ready"]).with_startup_timeout(30).with_poll_interval(0.01)

        with pytest.raises(ContainerExitedException, match=r"exit code 2"):
            strategy.wait_until_ready(target)
        assert target.exec_calls == 0

    def test_exec_strategy_timeout_text_unchanged_while_running(self):
        target = FakeWaitTarget(
            FakeWrappedContainer([_container_state("running")]),
            exec_result=SimpleNamespace(exit_code=1, output=b"not ready"),
        )
        strategy = ExecWaitStrategy(["check-ready"]).with_startup_timeout(1).with_poll_interval(0.01)

        with pytest.raises(TimeoutError, match=r"Command execution did not succeed within 1.000 seconds"):
            strategy.wait_until_ready(target)


class TestContainerStatusWaitStrategyExitHandling:
    """Status strategy used while resolving exposed ports."""

    def test_exited_container_fails_with_exit_code(self, monkeypatch: pytest.MonkeyPatch):
        target = FakeWaitTarget(FakeWrappedContainer([_container_state("exited", 1)]))
        strategy = ContainerStatusWaitStrategy().with_startup_timeout(1).with_poll_interval(0.01)
        monkeypatch.setattr(
            strategy, "get_status", Mock(side_effect=AssertionError("status must not be read for an exited container"))
        )

        with pytest.raises(ContainerExitedException, match=r"exit code 1"):
            strategy.wait_until_ready(target)

    def test_dead_container_fails_with_exit_code(self, monkeypatch: pytest.MonkeyPatch):
        target = FakeWaitTarget(FakeWrappedContainer([_container_state("dead", 139)]))
        strategy = ContainerStatusWaitStrategy().with_startup_timeout(1).with_poll_interval(0.01)
        monkeypatch.setattr(
            strategy, "get_status", Mock(side_effect=AssertionError("status must not be read for a dead container"))
        )

        with pytest.raises(ContainerExitedException, match=r"exit code 139"):
            strategy.wait_until_ready(target)

    def test_created_then_running_succeeds(self, monkeypatch: pytest.MonkeyPatch):
        target = FakeWaitTarget(
            FakeWrappedContainer(
                [_container_state("created"), _container_state("created"), _container_state("running")]
            )
        )
        strategy = ContainerStatusWaitStrategy().with_startup_timeout(1).with_poll_interval(0.01)
        monkeypatch.setattr(
            strategy, "get_status", lambda container: container.get_wrapped_container().attrs["State"]["Status"]
        )

        strategy.wait_until_ready(target)

    def test_paused_is_not_an_exit_and_times_out_with_original_text(self, monkeypatch: pytest.MonkeyPatch):
        target = FakeWaitTarget(FakeWrappedContainer([_container_state("paused")]))
        strategy = ContainerStatusWaitStrategy().with_startup_timeout(1).with_poll_interval(0.01)
        monkeypatch.setattr(strategy, "get_status", lambda container: "paused")

        with pytest.raises(TimeoutError, match=r"container did not become running"):
            strategy.wait_until_ready(target)

    def test_unreadable_status_is_not_an_exit(self, monkeypatch: pytest.MonkeyPatch):
        target = FakeWaitTarget(FakeWrappedContainer([_container_state("running")]))
        strategy = ContainerStatusWaitStrategy().with_startup_timeout(1).with_poll_interval(0.01)
        monkeypatch.setattr(strategy, "get_status", Mock(side_effect=RuntimeError("docker daemon unavailable")))

        with pytest.raises(TimeoutError, match=r"container did not become running"):
            strategy.wait_until_ready(target)


class TestComposeContainerReload:
    """Compose snapshots must observe service exits during waiting (ps always reports fresh)."""

    @staticmethod
    def _make_compose_mock(
        fresh: Optional[ComposeContainer], *, inspect_info: Optional[ContainerInspectInfo] = None
    ) -> tuple[Mock, Mock]:
        docker_compose = Mock()
        docker_compose.get_container.return_value = fresh
        docker_client = Mock()
        docker_client.get_container_inspect_info.return_value = inspect_info
        docker_compose._get_docker_client.return_value = docker_client
        return docker_compose, docker_client

    def test_reload_refreshes_exited_state_and_oom(self):
        fresh = ComposeContainer(Service="svc", State="exited", ExitCode=137, ID="abc")
        docker_compose, docker_client = self._make_compose_mock(
            fresh, inspect_info=ContainerInspectInfo(State=ContainerState(OOMKilled=True))
        )
        snapshot = ComposeContainer(Service="svc", State="running", ID="abc")
        snapshot._docker_compose = cast("Any", docker_compose)

        snapshot.reload()

        docker_compose.get_container.assert_called_once_with("svc", include_all=True)
        assert snapshot.State == "exited"
        assert snapshot.ExitCode == 137
        assert snapshot.OOMKilled is True

    def test_reload_keeps_last_snapshot_when_state_unreadable(self):
        docker_compose, docker_client = self._make_compose_mock(None)
        docker_compose.get_container.side_effect = RuntimeError("ps failed")
        snapshot = ComposeContainer(Service="svc", State="running", ID="abc")
        snapshot._docker_compose = cast("Any", docker_compose)

        snapshot.reload()

        assert snapshot.State == "running"
        assert snapshot.ExitCode is None
        docker_client.get_container_inspect_info.assert_not_called()

    def test_reload_does_not_inspect_oom_for_running_service(self):
        fresh = ComposeContainer(Service="svc", State="running", ID="abc")
        docker_compose, docker_client = self._make_compose_mock(
            fresh, inspect_info=ContainerInspectInfo(State=ContainerState(OOMKilled=True))
        )
        snapshot = ComposeContainer(Service="svc", State="running", ID="abc")
        snapshot._docker_compose = cast("Any", docker_compose)

        snapshot.reload()

        assert snapshot.State == "running"
        assert snapshot.OOMKilled is None
        docker_client.get_container_inspect_info.assert_not_called()

    def test_exit_detection_uses_refreshed_compose_snapshot(self):
        fresh = ComposeContainer(Service="svc", State="exited", ExitCode=137, ID="abc")
        docker_compose, docker_client = self._make_compose_mock(
            fresh, inspect_info=ContainerInspectInfo(State=ContainerState(OOMKilled=True))
        )
        snapshot = ComposeContainer(Service="svc", State="running", ID="abc")
        snapshot._docker_compose = cast("Any", docker_compose)

        exit_state = get_container_exit_state(snapshot)

        assert exit_state is not None
        assert exit_state.exit_code == 137
        assert exit_state.oom_killed is True
