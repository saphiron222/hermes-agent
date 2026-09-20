"""Reprise automatique des blocages temporaires et vérifiables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_block_recheck as recheck


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _running_task(conn, title: str) -> str:
    task_id = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
    assert kb.claim_task(conn, task_id, claimer="worker") is not None
    return task_id


def test_transient_waits_outside_human_bucket_then_retries_and_escalates(
    kanban_home: Path,
) -> None:
    with kbc.connect_closing() as conn:
        task_id = _running_task(conn, "temporary outage")

        assert kb.block_task(
            conn,
            task_id,
            reason="remote host restarting",
            kind="transient",
            retry_after=1_000,
        )
        parked = kb.get_task(conn, task_id)
        assert parked.status == "scheduled"
        assert parked.retry_after == 1_000
        assert task_id not in {task.id for task in kb.list_tasks(conn, status="blocked")}
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert (run.status, run.outcome) == ("scheduled", "scheduled")

        assert recheck.reevaluate_blocked_tasks(conn, now=999) == []
        resumed = recheck.reevaluate_blocked_tasks(conn, now=1_000)
        assert [item.task_id for item in resumed] == [task_id]
        assert kb.get_task(conn, task_id).status == "ready"

        assert kb.claim_task(conn, task_id, claimer="worker") is not None
        assert kb.block_task(
            conn,
            task_id,
            reason="remote host restarting",
            kind="transient",
            retry_after=2_000,
        )
        assert kb.get_task(conn, task_id).status == "triage"


def test_resume_check_is_fail_closed_and_records_passing_measurement(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with kbc.connect_closing() as conn:
        task_id = _running_task(conn, "wait for machine-readable condition")
        failing = {
            "type": "command",
            "argv": [sys.executable, "-c", "raise SystemExit(7)"],
            "timeout_seconds": 5,
        }
        assert kb.block_task(
            conn,
            task_id,
            reason="service unavailable",
            kind="capability",
            resume_check=failing,
        )

        assert recheck.reevaluate_blocked_tasks(conn, now=1_000) == []
        assert kb.get_task(conn, task_id).status == "blocked"
        assert not [c for c in kb.list_comments(conn, task_id) if c.author == "hermes-recheck"]

        monkeypatch.setenv("HERMES_KANBAN_TASK", "must-not-reach-check")
        passing = {
            "type": "command",
            "argv": [
                sys.executable,
                "-c",
                "import os; print('service-ready') if 'HERMES_KANBAN_TASK' not in os.environ else raise_error",
            ],
            "timeout_seconds": 5,
        }
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET resume_check = ?, retry_after = ? WHERE id = ?",
                (json.dumps(recheck.normalize_resume_check(passing), sort_keys=True), 1_000, task_id),
            )
        resumed = recheck.reevaluate_blocked_tasks(conn, now=1_000)

        assert [item.task_id for item in resumed] == [task_id]
        assert resumed[0].measurement == "command exit=0 stdout=service-ready"
        assert kb.get_task(conn, task_id).status == "ready"
        comments = [c.body for c in kb.list_comments(conn, task_id) if c.author == "hermes-recheck"]
        assert comments == ["REPRISE AUTOMATIQUE · command exit=0 stdout=service-ready"]


def test_default_backoff_and_human_block_without_check_stays_sticky(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recheck.time, "time", lambda: 10_000)
    with kbc.connect_closing() as conn:
        transient = _running_task(conn, "automatic backoff")
        assert kb.block_task(conn, transient, reason="temporary", kind="transient")
        parked = kb.get_task(conn, transient)
        assert parked is not None
        assert parked.status == "scheduled"
        assert parked.retry_after == 10_000 + recheck.DEFAULT_TRANSIENT_RETRY_SECONDS

        human = _running_task(conn, "human decision")
        assert kb.block_task(conn, human, reason="choose policy", kind="needs_input")
        sticky = kb.get_task(conn, human)
        assert sticky is not None
        assert sticky.status == "blocked"
        assert sticky.retry_after is None
        resumed = recheck.reevaluate_blocked_tasks(conn, now=99_999)
        assert [item.task_id for item in resumed] == [transient]
        still_sticky = kb.get_task(conn, human)
        assert still_sticky is not None
        assert still_sticky.status == "blocked"


def test_failed_check_is_rescheduled_and_invalid_payload_fails_closed(
    kanban_home: Path,
) -> None:
    with kbc.connect_closing() as conn:
        task_id = _running_task(conn, "failing machine check")
        assert kb.block_task(
            conn,
            task_id,
            reason="still unavailable",
            kind="capability",
            retry_after=1_000,
            resume_check={
                "type": "command",
                "argv": [sys.executable, "-c", "raise SystemExit(4)"],
            },
        )
        assert recheck.reevaluate_blocked_tasks(conn, now=1_000) == []
        failed = kb.get_task(conn, task_id)
        assert failed is not None
        assert failed.status == "blocked"
        assert failed.retry_after == 1_000 + recheck.RECHECK_FAILURE_DELAY_SECONDS

        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET resume_check = ?, retry_after = ? WHERE id = ?",
                ('{"type":"unknown"}', 2_000, task_id),
            )
        assert recheck.reevaluate_blocked_tasks(conn, now=2_000) == []
        invalid = kb.get_task(conn, task_id)
        assert invalid is not None
        assert invalid.status == "blocked"
        assert invalid.retry_after == 2_000 + recheck.RECHECK_FAILURE_DELAY_SECONDS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("30s", 1_030), ("2m", 1_120), ("1970-01-01T00:20:00Z", 1_200), (1_300, 1_300)],
)
def test_retry_after_accepts_duration_and_timestamps(raw, expected) -> None:
    assert recheck.parse_retry_after(raw, now=1_000) == expected
