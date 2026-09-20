"""Machine-verifiable recovery for parked Kanban tasks.

The scheduler-facing entry point is deliberately silent unless it resumes a
card.  Checks run without worker authority, fail closed, and are compared
against the stored condition before the state transition is committed.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_TRANSIENT_RETRY_SECONDS = 60
MAX_TRANSIENT_RETRY_SECONDS = 60 * 60
RECHECK_FAILURE_DELAY_SECONDS = 60
MAX_CHECK_TIMEOUT_SECONDS = 60
MAX_CHECKS_PER_RUN = 2
MAX_MEASUREMENT_CHARS = 300
_WORKER_ENV_PREFIX = "HERMES_KANBAN_"


@dataclass(frozen=True)
class ResumeResult:
    task_id: str
    measurement: str


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    measurement: str


def parse_retry_after(value: Any, *, now: Optional[int] = None) -> Optional[int]:
    """Normalize an epoch timestamp, ISO timestamp, or ``30s``/``5m``/``2h`` duration."""
    if value is None:
        return None
    current = int(time.time()) if now is None else int(now)
    if isinstance(value, bool):
        raise ValueError("retry_after must be a timestamp or duration, not a boolean")
    if isinstance(value, (int, float)):
        timestamp = int(value)
        if timestamp < 0:
            raise ValueError("retry_after timestamp must be non-negative")
        return timestamp
    if not isinstance(value, str) or not value.strip():
        raise ValueError("retry_after must be a Unix timestamp, ISO timestamp, or duration")
    raw = value.strip()
    if raw[-1:].lower() in {"s", "m", "h", "d"}:
        try:
            amount = float(raw[:-1])
        except ValueError as exc:
            raise ValueError(f"invalid retry_after duration: {value!r}") from exc
        if amount < 0:
            raise ValueError("retry_after duration must be non-negative")
        multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}[raw[-1].lower()]
        return current + int(amount * multiplier)
    if raw.isdigit():
        return int(raw)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid retry_after timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def transient_retry_at(recurrences: int, *, now: Optional[int] = None) -> int:
    """Exponential retry delay, capped at one hour, keyed to re-block cycles."""
    current = int(time.time()) if now is None else int(now)
    exponent = max(0, int(recurrences) - 1)
    delay = min(MAX_TRANSIENT_RETRY_SECONDS, DEFAULT_TRANSIENT_RETRY_SECONDS * (2 ** exponent))
    return current + delay


def normalize_resume_check(value: Any) -> Optional[dict[str, Any]]:
    """Validate and canonicalize a machine-verifiable resume condition."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("resume_check must be an object")
    check_type = value.get("type")
    timeout = value.get("timeout_seconds", 10)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("resume_check.timeout_seconds must be a number")
    timeout = float(timeout)
    if not 0 < timeout <= MAX_CHECK_TIMEOUT_SECONDS:
        raise ValueError(f"resume_check.timeout_seconds must be between 0 and {MAX_CHECK_TIMEOUT_SECONDS}")

    if check_type == "command":
        argv = value.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) and arg for arg in argv):
            raise ValueError("command resume_check.argv must be a non-empty string array")
        expected = value.get("expected_exit_code", 0)
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise ValueError("command resume_check.expected_exit_code must be an integer")
        return {
            "type": "command",
            "argv": list(argv),
            "expected_exit_code": expected,
            "timeout_seconds": timeout,
        }

    if check_type == "url":
        url = value.get("url")
        expected = value.get("expected_status", 200)
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url resume_check.url is required")
        if isinstance(expected, bool) or not isinstance(expected, int) or not 100 <= expected <= 599:
            raise ValueError("url resume_check.expected_status must be an HTTP status code")
        return {
            "type": "url",
            "url": url.strip(),
            "expected_status": expected,
            "timeout_seconds": timeout,
        }

    if check_type == "path_empty":
        path = value.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path_empty resume_check.path is required")
        return {"type": "path_empty", "path": path.strip(), "timeout_seconds": timeout}

    raise ValueError("resume_check.type must be one of: command, url, path_empty")


def serialize_resume_check(value: Any) -> Optional[str]:
    normalized = normalize_resume_check(value)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True) if normalized else None


def _clean_check_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith(_WORKER_ENV_PREFIX)}


def _one_line(value: str) -> str:
    from agent.redact import redact_sensitive_text

    cleaned = " ".join((value or "").split())
    cleaned = redact_sensitive_text(cleaned, force=True)
    if len(cleaned) > MAX_MEASUREMENT_CHARS:
        return cleaned[:MAX_MEASUREMENT_CHARS] + "…"
    return cleaned


def _run_command(check: dict[str, Any], workspace_path: Optional[str]) -> CheckResult:
    cwd = Path(workspace_path).expanduser() if workspace_path else None
    if cwd is not None and not cwd.is_dir():
        return CheckResult(False, "command not run: workspace is unavailable")
    try:
        completed = subprocess.run(
            check["argv"], cwd=str(cwd) if cwd else None, env=_clean_check_env(),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=check["timeout_seconds"], shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult(False, f"command failed: {_one_line(str(exc))}")
    output = _one_line(completed.stdout or completed.stderr)
    measurement = f"command exit={completed.returncode}"
    if output:
        measurement += f" stdout={output}"
    return CheckResult(completed.returncode == check["expected_exit_code"], measurement)


def _run_url(check: dict[str, Any]) -> CheckResult:
    from tools.url_safety import is_safe_url

    url = check["url"]
    if not is_safe_url(url):
        return CheckResult(False, "url blocked by SSRF protection")
    try:
        import httpx

        response = httpx.get(url, timeout=check["timeout_seconds"], follow_redirects=False)
    except Exception as exc:
        return CheckResult(False, f"url failed: {_one_line(str(exc))}")
    measurement = f"url status={response.status_code} expected={check['expected_status']}"
    return CheckResult(response.status_code == check["expected_status"], measurement)


def _run_path_empty(check: dict[str, Any], workspace_path: Optional[str]) -> CheckResult:
    path = Path(check["path"]).expanduser()
    if not path.is_absolute():
        if not workspace_path:
            return CheckResult(False, "path_empty has no workspace for its relative path")
        path = Path(workspace_path).expanduser() / path
    if not path.exists():
        return CheckResult(False, "path_empty target does not exist")
    try:
        empty = path.stat().st_size == 0 if path.is_file() else path.is_dir() and next(path.iterdir(), None) is None
    except OSError as exc:
        return CheckResult(False, f"path_empty failed: {_one_line(str(exc))}")
    return CheckResult(empty, f"path_empty empty={str(empty).lower()} path={_one_line(str(path))}")


def run_resume_check(check: dict[str, Any], *, workspace_path: Optional[str]) -> CheckResult:
    handlers = {
        "command": _run_command,
        "url": lambda item, _workspace: _run_url(item),
        "path_empty": _run_path_empty,
    }
    return handlers[check["type"]](check, workspace_path)


def _resume_if_unchanged(
    conn,
    *,
    task_id: str,
    expected_status: str,
    expected_retry_after: Optional[int],
    expected_resume_check: Optional[str],
    measurement: str,
    now: int,
) -> bool:
    from hermes_cli import kanban_db as kb

    with kb.write_txn(conn):
        row = conn.execute(
            "SELECT status, retry_after, resume_check FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if row is None or row["status"] != expected_status:
            return False
        if row["retry_after"] != expected_retry_after or row["resume_check"] != expected_resume_check:
            return False
        resume_status = kb._resume_status_from_events(conn, task_id)
        kb._reclaim_dangling_run(
            conn, task_id, statuses=("blocked", "scheduled"), now=now,
            note="invariant recovery on automatic resume",
        )
        landing_status = kb._landing_status_after_parents(conn, task_id)
        new_status = "review" if landing_status == "ready" and resume_status == "review" else landing_status
        updated = conn.execute(
            "UPDATE tasks SET status = ?, current_run_id = NULL, consecutive_failures = 0, "
            "last_failure_error = NULL, retry_after = NULL, resume_check = NULL "
            "WHERE id = ? AND status = ? AND retry_after IS ? AND resume_check IS ?",
            (new_status, task_id, expected_status, expected_retry_after, expected_resume_check),
        )
        if updated.rowcount != 1:
            return False
        body = f"REPRISE AUTOMATIQUE · {measurement}"
        kb._insert_comment(conn, task_id, "hermes-recheck", body, now)
        kb._append_event(
            conn, task_id, "unblocked",
            {"status": new_status, "resume_status": resume_status, "automatic": True, "measurement": measurement},
        )
        kb._append_event(conn, task_id, "commented", {"author": "hermes-recheck", "len": len(body)})
        return True


def reevaluate_blocked_tasks(conn, *, now: Optional[int] = None) -> list[ResumeResult]:
    """Run due checks and resume only conditions measured as passing."""
    current = int(time.time()) if now is None else int(now)
    rows = conn.execute(
        "SELECT id, status, workspace_path, retry_after, resume_check FROM tasks "
        "WHERE status IN ('blocked', 'scheduled') AND retry_after IS NOT NULL "
        "AND retry_after <= ? AND (status = 'scheduled' OR resume_check IS NOT NULL) "
        "ORDER BY retry_after, id LIMIT ?",
        (current, MAX_CHECKS_PER_RUN),
    ).fetchall()
    resumed: list[ResumeResult] = []
    for row in rows:
        serialized = row["resume_check"]
        if serialized:
            try:
                check = normalize_resume_check(json.loads(serialized))
                if check is None:
                    raise ValueError("resume_check cannot be null")
                result = run_resume_check(check, workspace_path=row["workspace_path"])
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                result = CheckResult(False, f"invalid resume_check: {_one_line(str(exc))}")
        else:
            result = CheckResult(True, f"retry_after reached ({row['retry_after']})")
        if not result.passed:
            from hermes_cli import kanban_db as kb

            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET retry_after = ? WHERE id = ? AND status = ? "
                    "AND retry_after IS ? AND resume_check IS ?",
                    (current + RECHECK_FAILURE_DELAY_SECONDS, row["id"], row["status"], row["retry_after"], serialized),
                )
            continue
        if _resume_if_unchanged(
            conn,
            task_id=row["id"],
            expected_status=row["status"],
            expected_retry_after=row["retry_after"],
            expected_resume_check=serialized,
            measurement=result.measurement,
            now=current,
        ):
            resumed.append(ResumeResult(row["id"], result.measurement))
    return resumed
