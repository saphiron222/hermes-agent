#!/usr/bin/env python3
"""Silent cron entry point for machine-verifiable Kanban recovery."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# A source checkout invocation has ``scripts/`` on sys.path, not the repository.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Cron recovery is an orchestrator action. Never carry a worker's task, run,
# claim, or board authority into this process; the profile's default board is
# resolved after the scrub.
for key in tuple(os.environ):
    if key.startswith("HERMES_KANBAN_"):
        os.environ.pop(key, None)

from hermes_cli import kanban_db as kb  # noqa: E402
from hermes_cli import kanban_db_connect as kbc  # noqa: E402
from hermes_cli.kanban_block_recheck import reevaluate_blocked_tasks  # noqa: E402


def main() -> int:
    kb.init_db()
    with kbc.connect_closing() as conn:
        resumed = reevaluate_blocked_tasks(conn)
    for item in resumed:
        print(f"Resumed {item.task_id}: {item.measurement}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
