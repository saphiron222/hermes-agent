"""Contain each managed Windows router tree without adopting the owner process."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import subprocess
import sys
import threading
import time

import psutil


class WindowsSpawnCleanupPending(RuntimeError):
    """A failed Windows spawn still owns live or unclosed kernel resources."""


_windows_spawn_lock = threading.RLock()
_pending_windows_cleanups = []


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _BasicAccounting(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


class _WindowsJob:
    def __init__(self):
        self._lock = threading.Lock()
        self._api = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, args, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            ("SetInformationJobObject", [wintypes.HANDLE, ctypes.c_int,
                                         ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ("QueryInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                           wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
            ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
        ):
            fn = getattr(self._api, name)
            fn.argtypes = args
            fn.restype = result
        # NULL security attributes create a non-inheritable, unnamed owner handle.
        self._handle = self._api.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = _ExtendedLimits()
            # Neither BREAKAWAY_OK nor SILENT_BREAKAWAY_OK: descendants stay contained.
            limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
            if not self._api.SetInformationJobObject(
                    self._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException as setup_error:
            try:
                self.close()
            except BaseException:
                cleanup = _WindowsSpawnCleanup(None, self, assigned=False)
                _retain_windows_cleanup(cleanup)
                raise WindowsSpawnCleanupPending(
                    f"{setup_error}; Windows Job cleanup remains pending"
                ) from setup_error
            raise

    def assign(self, proc: subprocess.Popen) -> None:
        # Popen retains the original process handle, avoiding a PID-reuse race.
        if not self._api.AssignProcessToJobObject(self._handle, int(proc._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        """Terminate the contained tree; repeated closes are harmless."""
        with self._lock:
            if self._handle is not None:
                if not self._api.CloseHandle(self._handle):
                    raise ctypes.WinError(ctypes.get_last_error())
                self._handle = None

    def terminate_and_wait(self, timeout: float = 10) -> bool:
        """Terminate every contained process and close only after proving extinction.

        A false result deliberately retains the Job handle: KILL_ON_JOB_CLOSE keeps
        containment fail-closed and a later dispatcher tick can retry the proof.
        """
        with self._lock:
            if self._handle is None:
                return True
            if not self._api.TerminateJobObject(self._handle, 1):
                return False
            deadline = time.monotonic() + timeout
            while True:
                accounting = _BasicAccounting()
                if not self._api.QueryInformationJobObject(
                        self._handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None):
                    return False
                if accounting.ActiveProcesses == 0:
                    if not self._api.CloseHandle(self._handle):
                        return False
                    self._handle = None
                    return True
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.01)


class _WindowsSpawnCleanup:
    """Retain process and Job authority until extinction and handle release are proven."""

    def __init__(self, proc, job: _WindowsJob, *, assigned: bool):
        self.proc = proc
        self.job = job
        self.assigned = assigned

    def terminate_and_wait(self, timeout: float = 10) -> bool:
        process_released = self.proc is None
        if self.proc is not None:
            process_released = True
            try:
                if self.proc.poll() is None:
                    self.proc.kill()
            except BaseException:
                process_released = False
            if process_released:
                try:
                    self.proc.wait(timeout=timeout)
                except BaseException:
                    process_released = False

        job_released = False
        try:
            if self.assigned:
                job_released = self.job.terminate_and_wait(timeout=timeout)
            else:
                self.job.close()
                job_released = True
        except BaseException:
            job_released = False

        if not process_released or not job_released:
            return False
        if self.proc is None:
            return True

        handles_released = True
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except BaseException:
                    handles_released = False
        try:
            self.proc._handle.Close()
        except BaseException:
            handles_released = False
        return handles_released


def _retain_windows_cleanup(cleanup: _WindowsSpawnCleanup) -> None:
    if cleanup not in _pending_windows_cleanups:
        _pending_windows_cleanups.append(cleanup)


def _retry_pending_windows_cleanups() -> bool:
    """Retry retained cleanup authorities; refuse new spawns while one is uncertain."""
    with _windows_spawn_lock:
        for cleanup in list(_pending_windows_cleanups):
            if cleanup.terminate_and_wait():
                _pending_windows_cleanups.remove(cleanup)
        return not _pending_windows_cleanups


def _finish_failed_windows_spawn(cleanup: _WindowsSpawnCleanup, cause: BaseException) -> None:
    if cleanup.terminate_and_wait():
        raise cause
    _retain_windows_cleanup(cleanup)
    raise WindowsSpawnCleanupPending(
        f"{cause}; Windows spawn cleanup remains pending"
    ) from cause


def spawn_server(cmd, **kwargs) -> tuple[subprocess.Popen, _WindowsJob | None]:
    """Start a router, returning its process and an owner-held containment handle.

    Keep the job until shutdown and call close() to terminate the entire tree.
    Windows closes it automatically if the owner dies. Other hosts retain Popen's
    ordinary behavior. Assignment happens before the child's first instruction.
    """
    if sys.platform != "win32":
        return subprocess.Popen(cmd, **kwargs), None
    with _windows_spawn_lock:
        if not _retry_pending_windows_cleanups():
            raise WindowsSpawnCleanupPending(
                "a previous Windows spawn cleanup remains pending"
            )
        job = _WindowsJob()
        proc = None
        assigned = False
        try:
            kwargs["creationflags"] = kwargs.get("creationflags", 0) | 0x00000004  # CREATE_SUSPENDED
            proc = subprocess.Popen(cmd, **kwargs)
            job.assign(proc)
            assigned = True
            psutil.Process(proc.pid).resume()
            return proc, job
        except WindowsSpawnCleanupPending:
            raise
        except BaseException as setup_error:
            _finish_failed_windows_spawn(
                _WindowsSpawnCleanup(proc, job, assigned=assigned), setup_error,
            )
