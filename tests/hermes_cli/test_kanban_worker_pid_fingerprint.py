"""A recycled worker PID is never mistaken for our worker.

``tasks.worker_pid`` survives a reboot; the number can then belong to an unrelated process. Every
liveness decision (extend/defer the claim) and every kill (SIGTERM/SIGKILL on timeout or reclaim)
must require the spawn-time start fingerprint to match, never bare PID existence.
"""

import os
import signal
import subprocess
import sys
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _claimed_running(conn, *, pid: int, started_at, max_runtime=None) -> str:
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=max_runtime)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = ?, started_at = ?, claim_expires = ? WHERE id = ?",
                     (started_at, old, old, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))
    return tid


def test_recycled_pid_is_reclaimed_without_being_signalled(board):
    """Our own live PID with a foreign fingerprint models a post-reboot recycle: the claim is released
    (dead worker), no signal is sent, and max-runtime enforcement does not SIGTERM the stranger either."""
    conn = board
    killed = []
    stranger_fingerprint = 1  # no live process started at tick 1
    tid = _claimed_running(conn, pid=os.getpid(), started_at=stranger_fingerprint, max_runtime=1)

    assert kbd._worker_alive(os.getpid(), stranger_fingerprint) is False
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.worker_pid is None

    tid2 = _claimed_running(conn, pid=os.getpid(), started_at=stranger_fingerprint)
    assert kb.release_stale_claims(conn, signal_fn=lambda pid, sig: killed.append((pid, sig))) == 1
    assert killed == []
    assert kb.get_task(conn, tid2).status == "ready"


def test_matching_fingerprint_keeps_the_live_worker_and_never_signals_caller(board):
    """The same PID with ITS OWN fingerprint (recorded at spawn) is our worker: the expired claim is
    extended rather than reclaimed, and cleanup cannot signal its own caller process."""
    from gateway.status import get_process_start_time

    conn = board
    killed = []
    tid = _claimed_running(conn, pid=os.getpid(), started_at=get_process_start_time(os.getpid()))
    assert kbd._worker_alive(os.getpid(), get_process_start_time(os.getpid())) is True
    assert kb.release_stale_claims(conn) == 0
    assert kb.get_task(conn, tid).status == "running"
    kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert "claim_extended" in kinds

    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET max_runtime_seconds = 1 WHERE id = ?", (tid,))
    assert kbd.enforce_max_runtime(
        conn, signal_fn=lambda pid, sig: killed.append((pid, sig))
    ) == []
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task is not None and task.status == "running"


def test_transient_fingerprint_probe_failure_holds_live_worker_claim(board, monkeypatch):
    """A temporary /proc/ps read failure is unknown, not proof that a verified worker died.

    Releasing the claim on that ambiguity starts a retry in the same workspace while the first
    worker still writes there.  The liveness and termination paths must both fail closed.
    """
    conn = board
    fingerprint = kbd._process_fingerprint(os.getpid())
    assert fingerprint is not None
    tid = _claimed_running(conn, pid=os.getpid(), started_at=fingerprint)
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda _pid: None)
    killed = []

    assert kbd._worker_alive(os.getpid(), fingerprint) is True
    assert kbd.detect_crashed_workers(conn) == []
    assert kb.release_stale_claims(
        conn, signal_fn=lambda pid, sig: killed.append((pid, sig))
    ) == 0
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task is not None and task.status == "running"


@pytest.mark.macos_only
def test_macos_ps_probe_failure_does_not_override_primary_pid_liveness(monkeypatch):
    """The optional zombie probe cannot turn a proven-live PID into a crash."""
    import gateway.status as status

    class FailedPsProbe:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(status, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(kbd.subprocess, "run", lambda *_args, **_kwargs: FailedPsProbe())

    assert kbd._pid_alive(424_242) is True


def test_max_runtime_holds_claim_until_verified_worker_really_exits(board, monkeypatch):
    """TERM/KILL delivery is not termination: a surviving worker keeps its workspace claim."""
    conn = board
    pid = 424_242
    fingerprint = "boot-a|123"
    tid = _claimed_running(conn, pid=pid, started_at=fingerprint, max_runtime=1)
    killed = []
    monkeypatch.setattr(kb, "_pid_alive", lambda candidate: candidate == pid)
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda candidate: fingerprint)
    monkeypatch.setattr(kbd, "_poll_worker_exit", lambda *_args, **_kwargs: False)

    assert kbd.enforce_max_runtime(
        conn, signal_fn=lambda candidate, sig: killed.append((candidate, sig))
    ) == []
    assert killed == [(pid, signal.SIGTERM), (pid, getattr(signal, "SIGKILL", signal.SIGTERM))]
    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "running"
    assert task.worker_pid == pid
    assert any(
        event.kind == "reclaim_deferred"
        and event.payload["reason"] == "max_runtime_worker_alive"
        for event in kb.list_events(conn, tid)
    )


def test_same_pid_and_start_tick_on_another_boot_is_foreign(board, monkeypatch):
    """A row that survived a reboot: the PID AND the boot-relative start tick both match a process on
    this boot (the Linux start time is clock ticks since boot, so that recurs), but the persisted
    instantiation epoch does not. The worker is foreign: claim released, zero signals."""
    from gateway import drain_control

    conn = board
    killed = []
    live_fingerprint = kbd._process_fingerprint(os.getpid())
    assert live_fingerprint is not None and live_fingerprint.split("|", 1)[1] == str(
        __import__("gateway.status", fromlist=["x"]).get_process_start_time(os.getpid()))
    tid = _claimed_running(conn, pid=os.getpid(), started_at=live_fingerprint, max_runtime=1)
    assert kbd._worker_alive(os.getpid(), live_fingerprint) is True

    # Same PID, same start tick, different boot identity.
    other_boot = "deadbeef-boot:1|" + live_fingerprint.split("|", 1)[1]
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = ? WHERE id = ?", (other_boot, tid))
    assert kbd._worker_alive(os.getpid(), other_boot) is False
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.worker_pid is None

    # The same value re-derived on THIS boot still identifies our worker (the witness is stable
    # within a boot, unlike the recorded epoch of a previous one).
    drain_control.current_instantiation_epoch.cache_clear()
    assert kbd._process_fingerprint(os.getpid()) == live_fingerprint


def test_unverified_fingerprint_capture_never_authorizes_a_signal(board, monkeypatch):
    """Fingerprint capture fails for a new spawn: the row is NOT a legacy NULL row. A live PID under
    it is never SIGTERM/SIGKILLed by any reclaim/timeout path, and the claim is held (not released
    beside the live process); once the PID is gone the claim is reclaimed normally."""
    import gateway.status as status

    conn = board
    killed = []
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=1)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    row = conn.execute("SELECT worker_started_at FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["worker_started_at"] == kbd.UNVERIFIED_WORKER_FINGERPRINT
    monkeypatch.undo()
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ?, claim_expires = ? WHERE id = ?", (old, old, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))

    sig = lambda pid, s: killed.append((pid, s))  # noqa: E731
    assert kbd.enforce_max_runtime(conn, signal_fn=sig) == []
    assert kb.release_stale_claims(conn, signal_fn=sig) == 0
    assert killed == []
    assert kb.get_task(conn, tid).status == "running"
    # An explicit operator reclaim releases the claim (human override) but still sends nothing.
    assert kb.reclaim_task(conn, tid, reason="operator", signal_fn=sig) is True
    assert killed == []

    # The process is gone (a dead PID): the row is reclaimed like any dead worker, still no signal.
    dead_pid = 424_243
    tid2 = kb.create_task(conn, title="job2", assignee="worker")
    kb.claim_task(conn, tid2)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_pid = ?, worker_started_at = ?, claim_expires = ? WHERE id = ?",
                     (dead_pid, kbd.UNVERIFIED_WORKER_FINGERPRINT, old, tid2))
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    assert kb.release_stale_claims(conn, signal_fn=sig) == 1
    assert killed == [] and kb.get_task(conn, tid2).status == "ready"


def _assert_dead_worker_group_is_gone_before_requeue(board, tmp_path):
    """A crash may leave a live terminal child; release only after the whole session is gone."""
    child_pid_file = tmp_path / "child.pid"
    script = (
        "import pathlib, subprocess, sys; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
        "sys.exit(7)"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", script, str(child_pid_file)],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not child_pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_pid_file.exists()
        child_pid = int(child_pid_file.read_text())
        deadline = time.monotonic() + 5
        while kbd._pid_alive(leader.pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert kbd._pid_alive(leader.pid) is False
        assert kbd._pid_alive(child_pid) is True

        fingerprint = kbd._process_fingerprint(leader.pid)
        assert fingerprint is not None  # zombie identity is readable until wait()/reap
        tid = _claimed_running(board, pid=leader.pid, started_at=fingerprint)

        assert kbd.detect_crashed_workers(board) == [tid]
        assert kb.get_task(board, tid).status == "ready"
        assert kbd._process_group_alive(leader.pid) is False
        assert kbd._pid_alive(child_pid) is False
    finally:
        with __import__("contextlib").suppress(ProcessLookupError):
            os.killpg(leader.pid, signal.SIGKILL)
        leader.wait(timeout=5)


@pytest.mark.linux_only
def test_linux_dead_worker_group_is_gone_before_requeue(board, tmp_path):
    _assert_dead_worker_group_is_gone_before_requeue(board, tmp_path)


@pytest.mark.macos_only
def test_macos_dead_worker_group_is_gone_before_requeue(board, tmp_path):
    _assert_dead_worker_group_is_gone_before_requeue(board, tmp_path)


def test_windows_reclaim_requires_retained_job_extinction(monkeypatch):
    class ExitedProcess:
        returncode = 7

        @staticmethod
        def poll():
            return 7

    class Job:
        def __init__(self, extinct):
            self.extinct = extinct
            self.calls = 0

        def terminate_and_wait(self, timeout=10):
            self.calls += 1
            return self.extinct

    pid = 424_245
    proc = ExitedProcess()
    job = Job(extinct=False)
    monkeypatch.setattr(kbd, "_live_worker_procs", {pid: proc})
    monkeypatch.setattr(kbd, "_live_worker_jobs", {pid: job})
    info = {
        "prev_pid": pid, "host_local": True, "termination_attempted": False,
        "terminated": False, "sigkill": False,
    }

    assert kbd._terminate_windows_crashed_worker_tree(pid, "boot|123", info)["terminated"] is False
    assert job.calls == 1
    assert pid in kbd._live_worker_procs and pid in kbd._live_worker_jobs

    job.extinct = True
    info["termination_attempted"] = False
    assert kbd._terminate_windows_crashed_worker_tree(pid, "boot|123", info)["terminated"] is True
    assert job.calls == 2
    assert pid not in kbd._live_worker_procs and pid not in kbd._live_worker_jobs
    assert kbd._classify_worker_exit(pid) == ("nonzero_exit", 7)


@pytest.mark.windows_only
def test_windows_descendant_spawning_during_cleanup_is_gone_before_requeue(board, tmp_path):
    """The Job contains descendants atomically, even while one is still spawning children."""
    child_pids_file = tmp_path / "children.txt"
    spawner = tmp_path / "spawner.py"
    spawner.write_text(
        "import pathlib, subprocess, sys, time\n"
        "p = pathlib.Path(sys.argv[1])\n"
        "while True:\n"
        "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "    with p.open('a') as stream:\n"
        "        stream.write(str(child.pid) + '\\n')\n"
        "    time.sleep(.01)\n"
    )
    leader_script = tmp_path / "leader.py"
    leader_script.write_text(
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])\n"
        "sys.exit(7)\n"
    )
    leader = kbd._spawn_windows_worker(
        [sys.executable, str(leader_script), str(spawner), str(child_pids_file)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        fingerprint = kbd._process_fingerprint(leader.pid)
        assert fingerprint is not None
        deadline = time.monotonic() + 5
        while (not child_pids_file.exists() or len(child_pids_file.read_text().splitlines()) < 2) \
                and time.monotonic() < deadline:
            time.sleep(0.01)
        child_pids = [int(value) for value in child_pids_file.read_text().splitlines()]
        assert len(child_pids) >= 2
        leader.wait(timeout=5)
        tid = _claimed_running(board, pid=leader.pid, started_at=fingerprint)

        assert kbd.detect_crashed_workers(board) == [tid]
        assert kb.get_task(board, tid).status == "ready"
        assert all(kbd._pid_alive(child_pid) is False for child_pid in child_pids)
    finally:
        kbd._live_worker_procs.pop(leader.pid, None)
        job = kbd._live_worker_jobs.pop(leader.pid, None)
        if job is not None:
            job.terminate_and_wait()


def test_crash_reclaim_holds_claim_when_worker_group_cannot_be_proven_gone(board, monkeypatch):
    """An unverifiable surviving group blocks retry instead of creating a second worker."""
    pid = 424_244
    tid = _claimed_running(board, pid=pid, started_at="boot-a|123")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kbd, "_process_group_alive", lambda _pgid: True)
    monkeypatch.setattr(kbd, "_worker_identity_matches", lambda *_args: None)

    assert kbd.detect_crashed_workers(board) == []
    task = kb.get_task(board, tid)
    assert task is not None and task.status == "running"
    spawned = []
    for _ in range(3):
        result = kbd.dispatch_once(
            board, spawn_fn=lambda *_args, **_kwargs: spawned.append(True) or 999_999,
        )
        assert result.spawned == []
    assert spawned == []
    assert board.execute("SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (tid,)).fetchone()[0] == 1
    assert any(
        event.kind == "reclaim_deferred"
        and event.payload["reason"] == "crashed_worker_group_alive"
        for event in kb.list_events(board, tid)
    )
