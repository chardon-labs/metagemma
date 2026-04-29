from __future__ import annotations

import os
import resource
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Protocol

from sandbox_harness.types import JsonObject


class UnsupportedBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandLimits:
    cpu_seconds: int | None = None
    memory_bytes: int | None = None
    file_size_bytes: int | None = None
    process_count: int | None = None


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass(frozen=True)
class SandboxConfig:
    timeout_seconds: int = 30
    limits: CommandLimits = field(default_factory=CommandLimits)
    env: Mapping[str, str] = field(
        default_factory=lambda: {
            "HOME": "/workspace",
            "LANG": "C.UTF-8",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "TMPDIR": "/tmp",
        }
    )


class SandboxSession(Protocol):
    workspace: Path

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: PurePosixPath | str = PurePosixPath("/workspace"),
        env: Mapping[str, str] | None = None,
        timeout_seconds: int | None = None,
        limits: CommandLimits | None = None,
    ) -> CommandResult:
        pass

    def cleanup(self) -> None:
        pass


class SandboxBackend(Protocol):
    name: str

    def is_supported(self) -> bool:
        pass

    def create_session(
        self,
        *,
        initial_files: Mapping[str, str] | None = None,
        metadata: JsonObject | None = None,
    ) -> SandboxSession:
        pass


def _apply_limits(limits: CommandLimits) -> None:
    if limits.cpu_seconds is not None:
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
    if limits.memory_bytes is not None:
        resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
    if limits.file_size_bytes is not None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.file_size_bytes, limits.file_size_bytes))
    if limits.process_count is not None:
        resource.setrlimit(resource.RLIMIT_NPROC, (limits.process_count, limits.process_count))


def _write_initial_files(workspace: Path, initial_files: Mapping[str, str] | None) -> None:
    if initial_files is None:
        return
    for relative_path, content in initial_files.items():
        if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise ValueError(f"Initial file path must stay inside the workspace: {relative_path}")
        target = workspace / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _merged_env(base: Mapping[str, str], override: Mapping[str, str] | None) -> dict[str, str]:
    merged = dict(base)
    if override is not None:
        merged.update(override)
    return merged


def _run_subprocess(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    timeout_seconds: int,
    limits: CommandLimits,
) -> CommandResult:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(argv),
            env=dict(env),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            preexec_fn=lambda: _apply_limits(limits),
            check=False,
        )
        return CommandResult(
            argv=tuple(argv),
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            timed_out=False,
            duration_seconds=time.monotonic() - started,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return CommandResult(
            argv=tuple(argv),
            exit_code=124,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
            duration_seconds=time.monotonic() - started,
        )


class _BaseSession:
    def __init__(self, *, workspace: Path, config: SandboxConfig) -> None:
        self.workspace = workspace
        self._config = config
        self._cleaned = False

    def cleanup(self) -> None:
        if self._cleaned:
            return
        shutil.rmtree(self.workspace, ignore_errors=True)
        self._cleaned = True

    def __enter__(self) -> _BaseSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.cleanup()

    def _timeout(self, timeout_seconds: int | None) -> int:
        return self._config.timeout_seconds if timeout_seconds is None else timeout_seconds

    def _limits(self, limits: CommandLimits | None) -> CommandLimits:
        return self._config.limits if limits is None else limits

    def _env(self, env: Mapping[str, str] | None) -> dict[str, str]:
        return _merged_env(self._config.env, env)


class BubblewrapSession(_BaseSession):
    def __init__(
        self,
        *,
        workspace: Path,
        config: SandboxConfig,
        bwrap_path: str,
        rootfs: Path,
        mount_path: PurePosixPath,
    ) -> None:
        super().__init__(workspace=workspace, config=config)
        self._bwrap_path = bwrap_path
        self._rootfs = rootfs
        self._mount_path = mount_path

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: PurePosixPath | str = PurePosixPath("/workspace"),
        env: Mapping[str, str] | None = None,
        timeout_seconds: int | None = None,
        limits: CommandLimits | None = None,
    ) -> CommandResult:
        cwd_path = PurePosixPath(cwd)
        command = [
            self._bwrap_path,
            "--die-with-parent",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-net",
            "--ro-bind",
            str(self._rootfs),
            "/",
            "--tmpfs",
            "/tmp",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--dir",
            str(self._mount_path),
            "--bind",
            str(self.workspace),
            str(self._mount_path),
            "--chdir",
            str(cwd_path),
            *argv,
        ]
        return _run_subprocess(
            command,
            env=self._env(env),
            timeout_seconds=self._timeout(timeout_seconds),
            limits=self._limits(limits),
        )


class BubblewrapBackend:
    name = "bubblewrap"

    def __init__(
        self,
        *,
        bwrap_path: str = "bwrap",
        rootfs: Path = Path("/"),
        mount_path: PurePosixPath = PurePosixPath("/workspace"),
        config: SandboxConfig | None = None,
    ) -> None:
        self.bwrap_path = bwrap_path
        self.rootfs = rootfs
        self.mount_path = mount_path
        self.config = SandboxConfig() if config is None else config

    def is_supported(self) -> bool:
        if shutil.which(self.bwrap_path) is None:
            return False
        probe = subprocess.run(
            [
                self.bwrap_path,
                "--die-with-parent",
                "--unshare-user",
                "--unshare-pid",
                "--ro-bind",
                "/",
                "/",
                "true",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return probe.returncode == 0

    def create_session(
        self,
        *,
        initial_files: Mapping[str, str] | None = None,
        metadata: JsonObject | None = None,
    ) -> SandboxSession:
        if not self.is_supported():
            raise UnsupportedBackendError(
                "Bubblewrap is unavailable or namespace creation is blocked. "
                "Install bwrap or use a host/container runtime that permits user and PID namespaces."
            )
        workspace = Path(tempfile.mkdtemp(prefix="sandbox-harness-bwrap-")).resolve()
        _write_initial_files(workspace, initial_files)
        return BubblewrapSession(
            workspace=workspace,
            config=self.config,
            bwrap_path=self.bwrap_path,
            rootfs=self.rootfs,
            mount_path=self.mount_path,
        )


class ProotSession(_BaseSession):
    def __init__(
        self,
        *,
        workspace: Path,
        config: SandboxConfig,
        proot_path: str,
        rootfs: Path,
        mount_path: PurePosixPath,
    ) -> None:
        super().__init__(workspace=workspace, config=config)
        self._proot_path = proot_path
        self._rootfs = rootfs
        self._mount_path = mount_path

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: PurePosixPath | str = PurePosixPath("/workspace"),
        env: Mapping[str, str] | None = None,
        timeout_seconds: int | None = None,
        limits: CommandLimits | None = None,
    ) -> CommandResult:
        command = [
            self._proot_path,
            "-r",
            str(self._rootfs),
            "-b",
            f"{self.workspace}:{self._mount_path}",
            "-w",
            str(PurePosixPath(cwd)),
            "--kill-on-exit",
            *argv,
        ]
        return _run_subprocess(
            command,
            env=self._env(env),
            timeout_seconds=self._timeout(timeout_seconds),
            limits=self._limits(limits),
        )


class ProotBackend:
    name = "proot"

    def __init__(
        self,
        *,
        proot_path: str = "proot",
        rootfs: Path = Path("/"),
        mount_path: PurePosixPath = PurePosixPath("/workspace"),
        config: SandboxConfig | None = None,
    ) -> None:
        self.proot_path = proot_path
        self.rootfs = rootfs
        self.mount_path = mount_path
        self.config = SandboxConfig() if config is None else config

    def is_supported(self) -> bool:
        if shutil.which(self.proot_path) is None:
            return False
        probe = subprocess.run(
            [self.proot_path, "-r", str(self.rootfs), "true"],
            capture_output=True,
            text=True,
            check=False,
        )
        return probe.returncode == 0

    def create_session(
        self,
        *,
        initial_files: Mapping[str, str] | None = None,
        metadata: JsonObject | None = None,
    ) -> SandboxSession:
        if not self.is_supported():
            raise UnsupportedBackendError(
                "PRoot is unavailable or ptrace is blocked. Install proot or run in an environment "
                "that permits ptrace for the selected backend."
            )
        workspace = Path(tempfile.mkdtemp(prefix="sandbox-harness-proot-")).resolve()
        _write_initial_files(workspace, initial_files)
        return ProotSession(
            workspace=workspace,
            config=self.config,
            proot_path=self.proot_path,
            rootfs=self.rootfs,
            mount_path=self.mount_path,
        )


def current_platform_note() -> str:
    return f"os={os.name} cwd={Path.cwd()}"
