"""Cron sessions must not inherit a kanban worker's dispatcher identity.

A cron job can be fired *in-process* from a kanban worker: the worker is a
normal ``hermes chat -q`` CLI agent (its default toolset includes ``cronjob``)
running with ``HERMES_KANBAN_TASK`` legitimately set in its own environment,
and ``cronjob(action="run")`` calls ``run_one_job()`` -> ``run_job()`` in that
same process.

Without isolation the cron ``AIAgent`` is misidentified as that worker: the
kanban toolset is force-added, the kanban-worker protocol is injected into its
system prompt, and ``kanban_complete`` defaults ``task_id`` to
``$HERMES_KANBAN_TASK`` — letting an unrelated cron job close the worker's task
and overwrite real results.

The in-process boundary is a **ContextVar**, deliberately not an
``os.environ`` clear: ``os.environ`` is process-global and shared with

  * the worker's own claim heartbeat (``run_agent._touch_activity`` ->
    ``heartbeat_current_worker_from_env``), which would starve and let the
    dispatcher reclaim a task whose worker is still alive;
  * the gateway's kanban watchers, which do their own board save/restore;
  * concurrent cron jobs on the parallel pool, which take a *shared* read lock
    and can interleave one another's snapshot/restore.

Real descendants use the upstream path fence instead: worker identity keys are
removed, board-location keys remain available for reads, and
``HERMES_DELEGATED_CHILD_CONTEXT`` carries the board root whose writes are
denied.  These tests assert the property (no inherited mutation authority), not
the obsolete implementation detail that every ``HERMES_KANBAN_*`` key vanishes.
"""

from __future__ import annotations

import ast
import json
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clear_kanban_detect_cache():
    """`_detect_environment` memoizes per process; kanban is context-dependent."""
    import agent.skill_utils as su

    su._ENV_DETECT_CACHE.pop("kanban", None)
    yield
    su._ENV_DETECT_CACHE.pop("kanban", None)


@pytest.fixture()
def worker_env(monkeypatch):
    """Simulate running inside a dispatcher-spawned kanban worker."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker_real_task")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", "/tmp/ws")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "lock-abc")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "team-alpha")


# ---------------------------------------------------------------------------
# The predicate itself
# ---------------------------------------------------------------------------

class TestDispatcherOwnedPredicate:
    def test_default_is_dispatcher_owned(self):
        from agent.delegation_context import is_dispatcher_owned_worker_context

        assert is_dispatcher_owned_worker_context() is True

    def test_false_inside_non_dispatcher_context(self):
        from agent.delegation_context import (
            is_dispatcher_owned_worker_context,
            non_dispatcher_owned_context,
        )

        with non_dispatcher_owned_context():
            assert is_dispatcher_owned_worker_context() is False
        assert is_dispatcher_owned_worker_context() is True

    def test_token_form_restores(self):
        from agent.delegation_context import (
            enter_non_dispatcher_owned_context,
            exit_non_dispatcher_owned_context,
            is_dispatcher_owned_worker_context,
        )

        token = enter_non_dispatcher_owned_context()
        assert is_dispatcher_owned_worker_context() is False
        exit_non_dispatcher_owned_context(token)
        assert is_dispatcher_owned_worker_context() is True

    def test_nesting_restores_outer_value(self):
        from agent.delegation_context import (
            is_dispatcher_owned_worker_context,
            non_dispatcher_owned_context,
        )

        with non_dispatcher_owned_context():
            with non_dispatcher_owned_context():
                assert is_dispatcher_owned_worker_context() is False
            assert is_dispatcher_owned_worker_context() is False
        assert is_dispatcher_owned_worker_context() is True

    def test_delegated_child_still_not_dispatcher_owned(self, monkeypatch):
        """The pre-existing delegate_task flag keeps its meaning."""
        import agent.delegation_context as dc

        token = dc._DELEGATED_CHILD_CONTEXT.set(True)
        try:
            assert dc.is_dispatcher_owned_worker_context() is False
        finally:
            dc._DELEGATED_CHILD_CONTEXT.reset(token)

    def test_thread_isolation(self, worker_env):
        """A ContextVar set in one thread must not leak into a sibling thread.

        This is the property an os.environ clear cannot provide, and the reason
        concurrent cron jobs can't corrupt each other.
        """
        from agent.delegation_context import (
            is_dispatcher_owned_worker_context,
            non_dispatcher_owned_context,
        )

        seen = {}
        release = threading.Event()

        def sibling():
            seen["sibling"] = is_dispatcher_owned_worker_context()
            release.set()

        def job():
            with non_dispatcher_owned_context():
                seen["job"] = is_dispatcher_owned_worker_context()
                t = threading.Thread(target=sibling)
                t.start()
                release.wait(5)
                t.join(5)

        t = threading.Thread(target=job)
        t.start()
        t.join(5)

        assert seen["job"] is False, "job thread must be marked non-dispatcher"
        assert seen["sibling"] is True, "sibling thread must be unaffected"


# ---------------------------------------------------------------------------
# The gates that consume it
# ---------------------------------------------------------------------------

class TestKanbanGatesRespectContext:
    def test_task_tools_hidden_from_cron_agent(self, worker_env):
        from agent.delegation_context import non_dispatcher_owned_context
        from tools import kanban_tools

        assert kanban_tools._check_kanban_mode() is True
        with non_dispatcher_owned_context():
            assert kanban_tools._check_kanban_mode() is False

    def test_profile_opt_in_keeps_schema_but_fences_parent_board(
        self, monkeypatch, worker_env, tmp_path
    ):
        """Schema visibility is not authority: the parent board stays fenced."""
        from agent.delegation_context import (
            kanban_path_is_fenced,
            non_dispatcher_owned_context,
        )
        from tools import kanban_tools

        parent_home = tmp_path / "parent-home"
        parent_home.mkdir()
        parent_db = parent_home / "kanban.db"
        scratch_db = tmp_path / "scratch" / "kanban.db"
        monkeypatch.setenv("HERMES_HOME", str(parent_home))
        monkeypatch.setenv("HERMES_KANBAN_DB", str(parent_db))
        monkeypatch.setattr(kanban_tools, "_profile_has_kanban_toolset", lambda: True)
        with non_dispatcher_owned_context():
            assert kanban_tools._check_kanban_mode() is True
            assert kanban_path_is_fenced(parent_db) is True
            assert kanban_path_is_fenced(scratch_db) is False

        monkeypatch.delenv("HERMES_KANBAN_TASK")
        with non_dispatcher_owned_context():
            assert kanban_path_is_fenced(parent_db) is False

    def test_cached_worker_tool_schema_cannot_mutate_from_cron(
        self, monkeypatch, worker_env, tmp_path
    ):
        """Even a stale/explicit schema cannot turn cron into the parent worker."""
        from types import SimpleNamespace

        from agent.delegation_context import non_dispatcher_owned_context
        from agent.agent_init import _load_tools
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        import model_tools
        from tools import kanban_tools
        from tools.registry import invalidate_check_fn_cache

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
        kb._INITIALIZED_PATHS.clear()
        kb.init_db()
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="parent", assignee="worker")
            comments_before = kb.list_comments(conn, tid)
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

        invalidate_check_fn_cache()
        model_tools._tool_defs_cache.clear()
        worker_tools = model_tools.get_tool_definitions(
            enabled_toolsets=["kanban"], quiet_mode=True
        )
        assert "kanban_show" in {
            tool["function"]["name"] for tool in worker_tools
        }

        with non_dispatcher_owned_context():
            cron_tools = model_tools.get_tool_definitions(
                enabled_toolsets=["kanban"], quiet_mode=True
            )
            cron_agent = SimpleNamespace(quiet_mode=True)
            _load_tools(cron_agent, ["kanban"], [])
            mutation = json.loads(kanban_tools._handle_comment({
                "task_id": tid,
                "body": "must not land",
            }))

        cron_names = {tool["function"]["name"] for tool in cron_tools}
        assert "kanban_comment" in cron_names
        assert cron_agent._kanban_worker_guidance == ""
        assert mutation.get("error")
        with kbc.connect() as conn:
            assert kb.list_comments(conn, tid) == comments_before

    def test_stop_nudge_hidden_from_cron_agent(self, worker_env):
        from agent.delegation_context import non_dispatcher_owned_context
        from agent.kanban_stop import kanban_stop_nudge_enabled

        assert kanban_stop_nudge_enabled() is True
        with non_dispatcher_owned_context():
            assert kanban_stop_nudge_enabled() is False

    def test_kanban_db_api_mutation_rejected_from_cron_agent(
        self, monkeypatch, worker_env, tmp_path
    ):
        """The Kanban DB API fence, not only the tool wrapper, preserves the parent."""
        from agent.delegation_context import non_dispatcher_owned_context
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
        kb._INITIALIZED_PATHS.clear()
        kb.init_db()
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="parent", assignee="worker")
            before = kb.list_comments(conn, tid)
            with non_dispatcher_owned_context(), pytest.raises(PermissionError):
                kb.add_comment(conn, tid, author="cron", body="must not land")
            assert kb.list_comments(conn, tid) == before

    def test_cron_child_env_carries_fence_and_drops_worker_identity(
        self, worker_env
    ):
        from agent.delegation_context import (
            DELEGATED_CHILD_ENV_MARKER,
            KANBAN_ENV_KEYS,
            non_dispatcher_owned_context,
        )
        from tools.environments.local import build_subprocess_env

        base = dict(os.environ)
        base["HERMES_HOME"] = "/tmp/profile-home"
        base["HERMES_KANBAN_DB"] = "/tmp/profile-home/kanban.db"
        with non_dispatcher_owned_context():
            child_env = build_subprocess_env(base=base)

        assert not (set(KANBAN_ENV_KEYS) & child_env.keys())
        assert child_env["HERMES_HOME"] == "/tmp/profile-home"
        assert child_env["HERMES_KANBAN_DB"] == "/tmp/profile-home/kanban.db"
        assert child_env[DELEGATED_CHILD_ENV_MARKER]

    def test_no_scrub_child_process_carries_effective_parent_board_fence(
        self, monkeypatch, worker_env
    ):
        """The no-secret-scrub spawn still drops identity and enforces the path fence."""
        from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER, KANBAN_ENV_KEYS
        from tools.environments.local import build_subprocess_env

        monkeypatch.setenv("HERMES_HOME", "/tmp/profile-home")
        monkeypatch.setenv("HERMES_KANBAN_DB", "/tmp/profile-home/kanban.db")
        monkeypatch.setenv("HERMES_KANBAN_FUTURE_CAPABILITY", "must-not-leak")
        monkeypatch.setenv("CRON_NO_SCRUB_SENTINEL", "must-survive")
        parent_before = dict(os.environ)

        child_env = build_subprocess_env(scrub_secrets=False)
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, os; "
                    "from agent.delegation_context import KANBAN_ENV_KEYS, kanban_path_is_fenced; "
                    "print(json.dumps({"
                    "'identity': sorted(k for k in KANBAN_ENV_KEYS if k in os.environ), "
                    "'db': os.environ.get('HERMES_KANBAN_DB'), "
                    "'marker': os.environ.get('HERMES_DELEGATED_CHILD_CONTEXT'), "
                    "'fenced': kanban_path_is_fenced(os.environ['HERMES_KANBAN_DB']), "
                    "'home': os.environ.get('HERMES_HOME'), "
                    "'sentinel': os.environ.get('CRON_NO_SCRUB_SENTINEL')}))"
                ),
            ],
            env=child_env,
            cwd=str(Path(__file__).resolve().parents[2]),
            check=True,
            capture_output=True,
            text=True,
        )
        observed = json.loads(probe.stdout)

        assert observed == {
            "identity": [],
            "db": "/tmp/profile-home/kanban.db",
            "marker": child_env[DELEGATED_CHILD_ENV_MARKER],
            "fenced": True,
            "home": "/tmp/profile-home",
            "sentinel": "must-survive",
        }
        assert not (set(KANBAN_ENV_KEYS) & child_env.keys())
        assert dict(os.environ) == parent_before

    def test_complete_does_not_default_to_worker_task(self, worker_env):
        """The damage path: kanban_complete must not inherit the task id."""
        from agent.delegation_context import non_dispatcher_owned_context
        from tools import kanban_tools

        assert kanban_tools._default_task_id(None) == "t_worker_real_task"
        with non_dispatcher_owned_context():
            assert kanban_tools._default_task_id(None) is None

    def test_explicit_task_id_still_honoured(self, worker_env):
        """Only the implicit default is suppressed, not an explicit argument."""
        from agent.delegation_context import non_dispatcher_owned_context
        from tools import kanban_tools

        with non_dispatcher_owned_context():
            assert kanban_tools._default_task_id("t_explicit") == "t_explicit"

    def test_skill_environment_gate(self, worker_env):
        from agent.delegation_context import non_dispatcher_owned_context
        import agent.skill_utils as su

        su._ENV_DETECT_CACHE.pop("kanban", None)
        assert su._detect_environment("kanban") is True
        with non_dispatcher_owned_context():
            su._ENV_DETECT_CACHE.pop("kanban", None)
            assert su._detect_environment("kanban") is False

    def test_kanban_env_verdict_is_not_memoized(self, worker_env):
        """`kanban` must bypass _ENV_DETECT_CACHE: caching it process-wide would
        freeze whichever context asked first and leak it to the others."""
        from agent.delegation_context import non_dispatcher_owned_context
        import agent.skill_utils as su

        su._ENV_DETECT_CACHE.pop("kanban", None)
        assert su._detect_environment("kanban") is True
        with non_dispatcher_owned_context():
            # No manual cache clear here — the production code must not have
            # cached the previous True.
            assert su._detect_environment("kanban") is False
        assert su._detect_environment("kanban") is True

    def test_toolset_force_add_suppressed(self, worker_env):
        from agent.delegation_context import non_dispatcher_owned_context
        import model_tools

        assert model_tools._is_dispatcher_owned_worker() is True
        with non_dispatcher_owned_context():
            assert model_tools._is_dispatcher_owned_worker() is False


# ---------------------------------------------------------------------------
# run_job wiring
# ---------------------------------------------------------------------------

class TestRunJobKanbanIsolation:
    @staticmethod
    def _install_stubs(monkeypatch, observed: dict, agent_cls=None):
        import sys

        import cron.scheduler as sched
        from cron import scheduler_delivery as sched_delivery
        from agent.delegation_context import is_dispatcher_owned_worker_context

        class FakeAgent:
            def __init__(self, **kwargs):
                observed["dispatcher_owned_during_init"] = (
                    is_dispatcher_owned_worker_context()
                )
                observed["kanban_env_during_init"] = {
                    k: v for k, v in os.environ.items()
                    if k.startswith("HERMES_KANBAN_")
                }

            def run_conversation(self, *_a, **_kw):
                observed["dispatcher_owned_during_run"] = (
                    is_dispatcher_owned_worker_context()
                )
                return {"final_response": "done", "messages": []}

            def get_activity_summary(self):
                return {"seconds_since_activity": 0.0}

        fake_mod = type(sys)("run_agent")
        fake_mod.AIAgent = agent_cls or FakeAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_mod)

        from hermes_cli import runtime_provider as _rtp

        monkeypatch.setattr(
            _rtp, "resolve_runtime_provider",
            lambda **_kw: {
                "provider": "test", "api_key": "k",
                "base_url": "http://test.local",
                "api_mode": "chat_completions",
            },
        )
        monkeypatch.setattr(
            sched, "_build_job_prompt", lambda job, prerun_script=None, **kw: "hi"
        )
        monkeypatch.setattr(sched_delivery, "_resolve_origin", lambda job: None)
        monkeypatch.setattr(sched, "_resolve_delivery_target", lambda job: None)
        monkeypatch.setattr(
            sched, "_resolve_cron_enabled_toolsets", lambda job, cfg: None
        )
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")

        import dotenv

        monkeypatch.setattr(dotenv, "load_dotenv", lambda *_a, **_kw: True)

    @staticmethod
    def _job(job_id="kanban-iso"):
        return {
            "id": job_id, "name": "kanban-iso-job",
            "workdir": None, "schedule_display": "manual",
        }

    def test_agent_runs_as_non_dispatcher(self, monkeypatch, worker_env):
        import cron.scheduler as sched
        from cron import scheduler_delivery as sched_delivery

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        success, *_ = sched.run_job(self._job())
        assert success is True
        assert observed["dispatcher_owned_during_init"] is False
        assert observed["dispatcher_owned_during_run"] is False

    def test_environment_is_left_untouched(self, monkeypatch, worker_env):
        """The whole point of the ContextVar: os.environ must not be mutated, so
        the worker's claim heartbeat and the gateway watchers keep working."""
        import cron.scheduler as sched
        from cron import scheduler_delivery as sched_delivery

        before = {
            k: v for k, v in os.environ.items() if k.startswith("HERMES_KANBAN_")
        }
        assert before, "fixture should have populated kanban env"

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        success, *_ = sched.run_job(self._job())
        assert success is True

        # Untouched DURING the job (the heartbeat thread reads it concurrently)...
        assert observed["kanban_env_during_init"] == before
        # ...and after.
        after = {
            k: v for k, v in os.environ.items() if k.startswith("HERMES_KANBAN_")
        }
        assert after == before

    def test_context_reset_after_job(self, monkeypatch, worker_env):
        import cron.scheduler as sched
        from cron import scheduler_delivery as sched_delivery
        from agent.delegation_context import is_dispatcher_owned_worker_context

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        sched.run_job(self._job("kanban-iso-reset"))
        assert is_dispatcher_owned_worker_context() is True

    def test_context_reset_even_when_job_raises(self, monkeypatch, worker_env):
        import cron.scheduler as sched
        from cron import scheduler_delivery as sched_delivery
        from agent.delegation_context import is_dispatcher_owned_worker_context

        class ExplodingAgent:
            def __init__(self, **kwargs):
                pass

            def run_conversation(self, *_a, **_kw):
                raise RuntimeError("boom")

            def get_activity_summary(self):
                return {"seconds_since_activity": 0.0}

        observed: dict = {}
        self._install_stubs(monkeypatch, observed, agent_cls=ExplodingAgent)

        success, *_ = sched.run_job(self._job("kanban-iso-fail"))
        assert success is False
        assert is_dispatcher_owned_worker_context() is True
        # And the env survived the failure too.
        assert os.environ.get("HERMES_KANBAN_BOARD") == "team-alpha"

    def test_concurrent_jobs_do_not_corrupt_worker_identity(
        self, monkeypatch, worker_env
    ):
        """Two workdir-less jobs run concurrently on the parallel pool and take a
        SHARED read lock, so they interleave. With an os.environ snapshot/clear/
        restore this permanently destroyed the worker's identity; a ContextVar is
        per-thread and cannot."""
        import cron.scheduler as sched
        from cron import scheduler_delivery as sched_delivery

        before = {
            k: v for k, v in os.environ.items() if k.startswith("HERMES_KANBAN_")
        }
        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        results = {}

        def run(name):
            ok, *_ = sched.run_job(self._job(f"kanban-iso-{name}"))
            results[name] = ok

        threads = [threading.Thread(target=run, args=(n,)) for n in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)

        assert results == {"a": True, "b": True}
        after = {
            k: v for k, v in os.environ.items() if k.startswith("HERMES_KANBAN_")
        }
        assert after == before, "worker identity must survive concurrent cron jobs"


def test_registered_direct_cron_run_cannot_mutate_parent_board(
    tmp_path, monkeypatch, worker_env
):
    """Exercise the registered ``cronjob_manage(action=run)`` direct path.

    The in-process cron agent loses the worker-only Kanban surface.  A stale
    tool call, a direct DB mutation, and a terminal child invoking the Kanban
    CLI must all leave the parent board unchanged.  The child may retain the
    board location for reads, but it carries the upstream path fence and no
    dispatcher identity.
    """
    import cron.scheduler as sched
    from cron import jobs
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import cronjob_tools, kanban_tools  # noqa: F401 - register real handlers
    from tools.registry import registry

    home = tmp_path / "home"
    home.mkdir()
    cron_dir = home / "cron"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(home / "workspaces"))
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    monkeypatch.setattr(sched, "_launch_external_cron_worker", lambda _job: False)
    monkeypatch.setattr(cronjob_tools, "_try_dispatch_background_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(cronjob_tools, "_forward_relay_fronted_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "_open_cron_session_db", lambda _job: None)
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")

    kb.init_db()
    with kbc.connect_closing() as conn:
        sentinel = kb.create_task(conn, title="sentinel", assignee="dev")
        before_events = len(kb.list_events(conn, sentinel))

    observed: dict = {}
    parent_env = {
        key: value for key, value in os.environ.items()
        if key == "HERMES_HOME" or key.startswith("HERMES_KANBAN_")
    }
    root = Path(__file__).parents[2]

    class ProbeAgent:
        def __init__(self, **kwargs):
            import model_tools

            tools = model_tools.get_tool_definitions(
                enabled_toolsets=kwargs.get("enabled_toolsets"), quiet_mode=True
            )
            observed["tool_names"] = {
                item["function"]["name"] for item in tools
            }

        def run_conversation(self, *_args, **_kwargs):
            observed["tool_mutation"] = json.loads(
                registry.dispatch(
                    "kanban_complete",
                    {"task_id": sentinel, "summary": "must be refused"},
                )
            )
            observed["db_mutation_refused"] = False
            try:
                with kbc.connect_closing() as direct_conn:
                    kb.add_comment(
                        direct_conn, sentinel, author="cron", body="must be refused"
                    )
            except PermissionError as exc:
                observed["db_mutation_refused"] = True
                observed["db_mutation"] = str(exc)
            probe = (
                "import json, os, subprocess, sys; "
                "from agent.delegation_context import KANBAN_ENV_KEYS, kanban_path_is_fenced; "
                "print('IDENTITY=' + json.dumps(sorted(k for k in KANBAN_ENV_KEYS if k in os.environ))); "
                "print('DB=' + str(os.environ.get('HERMES_KANBAN_DB'))); "
                "print('FENCED=' + str(kanban_path_is_fenced(os.environ['HERMES_KANBAN_DB']))); "
                f"p=subprocess.run([sys.executable, '-m', 'hermes_cli.main', "
                f"'kanban', 'comment', {sentinel!r}, 'must be refused'], "
                "capture_output=True, text=True); "
                "print('RC=' + str(p.returncode)); print('ERR=' + p.stderr.strip())"
            )
            from tools.environments.local import LocalEnvironment

            env = LocalEnvironment(cwd=str(root), timeout=30)
            try:
                observed["terminal"] = env.execute(
                    f"{shlex.quote(sys.executable)} -c {shlex.quote(probe)}",
                    timeout=30,
                )
            finally:
                env.cleanup()
            return {
                "final_response": "probe complete",
                "messages": [{"role": "assistant", "content": "probe complete"}],
            }

        def get_activity_summary(self):
            return {"seconds_since_activity": 0.0}

    fake_run_agent = type(sys)("run_agent")
    fake_run_agent.AIAgent = ProbeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "test", "api_key": "k",
            "base_url": "http://test.local", "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr(sched, "_build_job_prompt", lambda *_args, **_kwargs: "probe")
    monkeypatch.setattr(sched, "_resolve_delivery_target", lambda _job: None)

    created = json.loads(registry.dispatch("cronjob_manage", {
        "action": "create", "name": "isolation probe", "schedule": "1h",
        "prompt": "Run the isolation probe", "deliver": "local",
    }))
    assert created.get("success") is True, created
    result = json.loads(registry.dispatch(
        "cronjob_manage", {"action": "run", "job_id": created["job_id"]}
    ))

    assert result["job"]["execution_success"] is True
    assert not any(name.startswith("kanban_") for name in observed["tool_names"])
    assert observed["tool_mutation"].get("error")
    assert observed["db_mutation_refused"] is True
    assert "cannot mutate Kanban" in observed["db_mutation"]
    terminal_output = observed["terminal"]["output"]
    assert "IDENTITY=[]" in terminal_output
    assert f"DB={home / 'kanban.db'}" in terminal_output
    assert "FENCED=True" in terminal_output
    assert "RC=1" in terminal_output
    assert "cannot mutate Kanban tasks via the CLI" in terminal_output
    assert {
        key: value for key, value in os.environ.items()
        if key == "HERMES_HOME" or key.startswith("HERMES_KANBAN_")
    } == parent_env

    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, sentinel).status == "ready"
        assert kb.list_comments(conn, sentinel) == []
        assert len(kb.list_events(conn, sentinel)) == before_events


@pytest.mark.linux_only
def test_dispatcher_grants_only_the_assigned_worker_scope(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import sys
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from hermes_cli.kanban_db_dispatch import _default_spawn

    monkeypatch.setenv("HOME", str(tmp_path))
    db = tmp_path / "board.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    conn = connect(db)
    tid = kb.create_task(conn, title="assigned child", assignee="default")
    kb.claim_task(conn, tid)
    task = kb.get_task(conn, tid)
    output = tmp_path / "worker-result.json"
    worker = tmp_path / "fixture-worker"
    root = str(Path(__file__).resolve().parents[2])
    worker.write_text(
        f"#!{sys.executable}\nimport sys, os, json;sys.path.insert(0, {root!r})\n"
        "from tools.kanban_tools import _handle_complete, heartbeat_current_worker_from_env\n"
        "beat=heartbeat_current_worker_from_env()\n"
        f"result=json.loads(_handle_complete({{'summary':'assigned worker'}}));result['beat']=beat\n"
        f"open({str(output)!r}, 'w').write(json.dumps(result))\n"
    )
    worker.chmod(0o700)
    monkeypatch.setenv("HERMES_BIN", str(worker))
    # Building a new worker under an existing task must replace, not inherit, its scope.
    monkeypatch.setenv("HERMES_KANBAN_TASK", "prior-task")
    # A dispatcher launched from an agent's shell carries the descendant fence itself; the worker it
    # grants a task to must not (an inherited marker fences the worker's own heartbeat + handoff).
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", str(tmp_path))
    pid = _default_spawn(task, str(tmp_path), board="default")
    assert pid is not None
    os.waitpid(pid, 0)  # windows-footgun: ok — Linux-only real dispatcher spawn
    result = json.loads(output.read_text())
    assert result["ok"] and result["beat"] is True, result
    assert kb.get_task(conn, tid).status == "done"
    assert os.environ["HERMES_KANBAN_TASK"] == "prior-task"
    conn.close()
