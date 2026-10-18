import time
from pathlib import Path

import pytest

from testcontainers.compose import DockerCompose
from testcontainers.core.container import DockerContainer
from testcontainers.core.exceptions import ContainerExitedException
from testcontainers.core.wait_strategies import (
    ExecWaitStrategy,
    FileExistsWaitStrategy,
    HttpWaitStrategy,
    LogMessageWaitStrategy,
    PortWaitStrategy,
)


class TestRealDockerIntegration:
    """Integration tests using real Docker containers."""

    def test_log_message_wait_strategy_with_real_container(self):
        """Test LogMessageWaitStrategy with a real container that outputs known logs."""
        strategy = LogMessageWaitStrategy("Hello from Docker!")

        with DockerContainer("hello-world").waiting_for(strategy) as container:
            # If we get here, the strategy worked
            assert container.get_wrapped_container() is not None

    def test_wait_strategy_timeout_with_real_container(self):
        """Test that wait strategies properly timeout with real containers."""
        # Use a very short timeout with a condition that won't be met
        strategy = LogMessageWaitStrategy("this_message_will_never_appear").with_startup_timeout(2)

        with pytest.raises(TimeoutError):
            with DockerContainer("alpine:latest").with_command("sleep 10").waiting_for(strategy):
                pass  # Should not reach here


class TestWaitStrategiesDetectContainerExit:
    """Wait strategies fail immediately (not after the timeout) once a container exits."""

    @staticmethod
    def _start_expecting_exit(
        container, expected_exit_code: int, *, expected_oom: bool = False, max_seconds: float = 30
    ):
        start = time.monotonic()
        try:
            with pytest.raises(ContainerExitedException) as exc_info:
                container.start()
            assert time.monotonic() - start < max_seconds
            message = str(exc_info.value)
            assert f"exit code {expected_exit_code}" in message
            assert ("OOM" in message) is expected_oom
            # Startup failures due to an exit must preserve the container logs, like timeouts.
            notes = getattr(exc_info.value, "__notes__", [])
            return notes
        finally:
            container.stop()

    @staticmethod
    def _wait_for_container_exit(container, *, timeout: float = 15) -> int:
        wrapped = container.get_wrapped_container()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            wrapped.reload()
            if wrapped.status == "exited":
                return int(wrapped.attrs["State"]["ExitCode"])
            time.sleep(0.05)
        raise AssertionError("container did not exit within the expected time")

    def test_http_strategy_reports_exit_code_when_container_crashes_while_waiting(self):
        strategy = HttpWaitStrategy(8080).for_status_code(200)
        container = (
            DockerContainer("alpine:latest")
            .with_exposed_ports(8080)
            .with_command(["sh", "-c", "echo http-crash-marker; sleep 1; exit 1"])
            .waiting_for(strategy)
        )

        notes = self._start_expecting_exit(container, 1)

        assert any("http-crash-marker" in note for note in notes), notes

    def test_http_strategy_reports_exit_code_zero_when_container_exits_before_wait(self):
        strategy = HttpWaitStrategy(8080).for_status_code(200)
        container = DockerContainer("alpine:latest").with_command(["sh", "-c", "exit 0"]).waiting_for(strategy)

        self._start_expecting_exit(container, 0, max_seconds=10)

    def test_http_strategy_reports_oom_when_container_killed_while_waiting(self):
        strategy = HttpWaitStrategy(8080).for_status_code(200)
        container = (
            DockerContainer("alpine:latest")
            .with_exposed_ports(8080)
            .with_tmpfs_mount("/tmp")
            .with_kwargs(mem_limit="8m", memswap_limit="8m")
            .with_command(["sh", "-c", "dd if=/dev/zero of=/tmp/f bs=1M count=64 status=none"])
            .waiting_for(strategy)
        )

        self._start_expecting_exit(container, 137, expected_oom=True)

    def test_wait_strategies_report_exit_code_when_container_already_exited(self):
        container = DockerContainer("alpine:latest").with_exposed_ports(8080).with_command(["sh", "-c", "exit 42"])
        container.start()
        try:
            assert self._wait_for_container_exit(container) == 42

            # HTTP/port strategies used to report "container did not become running"
            # without an exit code; command/file strategies waited for the full timeout.
            with pytest.raises(ContainerExitedException, match=r"exit code 42"):
                HttpWaitStrategy(8080).for_status_code(200).wait_until_ready(container)
            with pytest.raises(ContainerExitedException, match=r"exit code 42"):
                PortWaitStrategy(8080).wait_until_ready(container)
            with pytest.raises(ContainerExitedException, match=r"exit code 42"):
                ExecWaitStrategy(["false"]).wait_until_ready(container)
            with pytest.raises(ContainerExitedException, match=r"exit code 42"):
                FileExistsWaitStrategy("/tmp/marker").wait_until_ready(container)
        finally:
            container.stop()

    def test_exec_strategy_reports_exit_code_when_container_crashes_while_waiting(self):
        strategy = ExecWaitStrategy(["false"]).with_poll_interval(0.1)
        container = DockerContainer("alpine:latest").with_command(["sh", "-c", "sleep 1; exit 7"]).waiting_for(strategy)

        self._start_expecting_exit(container, 7)

    def test_http_strategy_timeout_is_unchanged_for_running_container(self):
        strategy = HttpWaitStrategy(8080).for_status_code(200).with_startup_timeout(1)
        container = (
            DockerContainer("alpine:latest").with_exposed_ports(8080).with_command("sleep 60").waiting_for(strategy)
        )
        try:
            with pytest.raises(TimeoutError, match=r"HTTP endpoint not ready within 1.0 seconds"):
                container.start()
        finally:
            container.stop()


class TestDockerComposeExitDetection:
    """Wait strategies observe fresh compose service state instead of a stale snapshot."""

    def test_compose_strategy_fails_fast_when_service_exits(self):

        compose = DockerCompose(
            context=Path(__file__).parent / "compose_fixtures" / "exiting_service",
            compose_file_name="compose.yaml",
            wait=False,
        )
        compose.waiting_for({"exiting": HttpWaitStrategy(8080).for_status_code(200).with_startup_timeout(30)})

        start = time.monotonic()
        try:
            with pytest.raises(ContainerExitedException, match=r"exit code 5"):
                compose.start()
            assert time.monotonic() - start < 20
        finally:
            compose.stop()

    def test_compose_strategy_timeout_is_unchanged_for_running_service(self):

        compose = DockerCompose(
            context=Path(__file__).parent / "compose_fixtures" / "basic",
            compose_file_name="docker-compose.yaml",
            wait=False,
        )
        compose.waiting_for({"alpine": HttpWaitStrategy(8080).for_status_code(200).with_startup_timeout(1)})

        try:
            with pytest.raises(TimeoutError, match=r"HTTP endpoint not ready within 1.0 seconds"):
                compose.start()
        finally:
            compose.stop()


class TestDockerComposeIntegration:
    """Integration tests for wait strategies with Docker Compose."""

    def test_compose_service_wait_strategies(self):
        """Test that wait strategies work with Docker Compose services."""
        from pathlib import Path

        # Use basic_multiple fixture with two alpine services that output logs
        compose = DockerCompose(
            context=Path(__file__).parent / "compose_fixtures" / "basic_multiple",
            compose_file_name="docker-compose.yaml",
        )

        # Configure wait strategies for both services
        # Wait for the date output that these containers produce
        compose.waiting_for(
            {
                "alpine1": LogMessageWaitStrategy("202").with_startup_timeout(30),  # Date includes year 202X
                "alpine2": LogMessageWaitStrategy("202").with_startup_timeout(30),  # Date includes year 202X
            }
        )

        with compose:
            # Verify both services are running
            container1 = compose.get_container("alpine1")
            container2 = compose.get_container("alpine2")

            assert container1.State == "running"
            assert container2.State == "running"

            # Verify logs contain expected patterns
            logs1 = container1.get_logs()
            logs2 = container2.get_logs()

            # Both containers should have date output (which contains "202" for year 202X)
            assert any(b"202" in log for log in logs1)
            assert any(b"202" in log for log in logs2)

    def test_compose_wait_strategy_timeout(self):
        """Test that compose wait strategies properly timeout."""

        compose = DockerCompose(
            context=Path(__file__).parent / "compose_fixtures" / "basic", compose_file_name="docker-compose.yaml"
        )

        # Use a wait strategy that will never succeed with very short timeout
        compose.waiting_for(
            {"alpine": LogMessageWaitStrategy("this_message_will_never_appear").with_startup_timeout(2)}
        )

        with pytest.raises(TimeoutError):
            with compose:
                pass  # Should not reach here
