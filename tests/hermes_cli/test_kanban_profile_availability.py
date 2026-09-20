"""Kanban profile readiness guards.

A profile directory is not proof that a worker can start: routing requires a
live gateway and usable credentials in that profile's isolated scope.
"""

from pathlib import Path
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import profile_availability
from hermes_cli.profiles import ProfileInfo


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_dispatcher_blocks_existing_card_for_unavailable_profile(
    kanban_home, monkeypatch, all_assignees_spawnable,
):
    """An existing assignment to a dead profile is blocked with the measured
    cause before Popen, without spending a retry."""
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="must not spin", assignee="dead")

        monkeypatch.setattr(
            profile_availability,
            "worker_availability_for_name",
            lambda *_args, **_kwargs: profile_availability.ProfileWorkerAvailability(
                False,
                "profile 'dead' has no usable credentials for provider 'openai-codex'",
            ),
        )
        monkeypatch.setattr(
            kbd,
            "_default_spawn",
            lambda *_args, **_kwargs: pytest.fail("unavailable profile reached Popen"),
        )

        result = kbd.dispatch_once(conn)
        task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)

    assert result.auto_blocked == [task_id]
    assert result.spawned == []
    assert task.status == "blocked"
    assert task.block_kind == "capability"
    assert task.consecutive_failures == 0
    blocked = [event for event in events if event.kind == "blocked"]
    assert blocked
    assert blocked[-1].payload["reason"] == (
        "assignee unavailable: profile 'dead' has no usable credentials "
        "for provider 'openai-codex'"
    )


def test_profile_worker_availability_requires_auth_and_preserves_healthy_profile(
    tmp_path, monkeypatch,
):
    profile = ProfileInfo(
        name="worker",
        path=tmp_path,
        is_default=False,
        gateway_running=True,
        provider="openai-codex",
        model="gpt-5.6-sol",
    )
    from hermes_cli import models

    profile.provider = ""
    assert "no provider configured" in profile_availability.profile_worker_availability(profile).reason
    profile.provider = "openai-codex"
    profile.model = ""
    assert "no model configured" in profile_availability.profile_worker_availability(profile).reason

    monkeypatch.setattr(models, "_provider_has_credentials", lambda _provider: False)
    unavailable = profile_availability.profile_worker_availability(
        profile,
        provider_override="openai-codex",
        model_override="gpt-5.6-sol",
    )
    assert unavailable.available is False
    assert "no usable credentials" in unavailable.reason

    monkeypatch.setattr(models, "_provider_has_credentials", lambda _provider: True)
    profile.model = "gpt-5.6-sol"
    available = profile_availability.profile_worker_availability(profile)
    assert available == profile_availability.ProfileWorkerAvailability(True)


def test_profile_worker_availability_accepts_configured_named_custom_provider(
    kanban_home, monkeypatch,
):
    profile_home = kanban_home / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "model:\n"
        "  provider: custom:my-gateway\n"
        "  default: gpt-5.4\n"
        "providers:\n"
        "  my-gateway:\n"
        "    base_url: https://gateway.example.com/v1\n"
        "    api_key: test-key\n",
        encoding="utf-8",
    )
    profile = ProfileInfo(
        name="worker",
        path=profile_home,
        is_default=False,
        gateway_running=True,
        provider="custom:my-gateway",
        model="gpt-5.4",
    )
    from hermes_cli import auth

    monkeypatch.setattr(
        auth,
        "get_auth_status",
        lambda provider: pytest.fail(f"named custom provider reached auth registry: {provider}"),
    )

    assert profile_availability.profile_worker_availability(profile) == (
        profile_availability.ProfileWorkerAvailability(True)
    )

    from hermes_cli import kanban_decompose, profiles

    monkeypatch.setattr(profiles, "list_profiles", lambda **_kwargs: [profile])
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "worker")
    roster, valid_names = kanban_decompose._build_roster()
    assert [entry["name"] for entry in roster] == ["worker"]
    assert valid_names == {"worker"}

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="custom route", assignee="worker")
        monkeypatch.setattr(kbd, "_default_spawn", lambda *_args, **_kwargs: 0)
        result = kbd.dispatch_once(conn)
        dispatched = kb.get_task(conn, task_id)
    assert [row[0] for row in result.spawned] == [task_id]
    assert dispatched is not None
    assert dispatched.status == "running"

    profile.provider = "custom:missing"
    unavailable = profile_availability.profile_worker_availability(profile)
    assert unavailable.available is False
    assert "no usable credentials" in unavailable.reason


@pytest.mark.parametrize("auth_error", [None, RuntimeError("auth failed")])
def test_profile_worker_availability_restores_a_after_reading_b(
    tmp_path, monkeypatch, auth_error,
):
    from agent.secret_scope import current_secret_scope, get_secret, reset_secret_scope, set_secret_scope
    from hermes_constants import (
        get_hermes_home_override,
        reset_hermes_home_override,
        set_hermes_home_override,
    )
    from hermes_cli import models

    home_a = tmp_path / "profiles" / "a"
    home_b = tmp_path / "profiles" / "b"
    home_a.mkdir(parents=True)
    home_b.mkdir(parents=True)
    (home_b / ".env").write_text("PROFILE_KEY=from-b\n", encoding="utf-8")
    profile = ProfileInfo(
        name="b",
        path=home_b,
        is_default=False,
        gateway_running=True,
        provider="openai-codex",
        model="gpt-5.6-sol",
    )
    scope_a = {"PROFILE_KEY": "from-a"}
    home_token = set_hermes_home_override(home_a)
    secret_token = set_secret_scope(scope_a)

    def inspect_b(_provider):
        assert get_hermes_home_override() == str(home_b)
        assert get_secret("PROFILE_KEY") == "from-b"
        if auth_error is not None:
            raise auth_error
        return True

    monkeypatch.setattr(models, "_provider_has_credentials", inspect_b)
    try:
        verdict = profile_availability.profile_worker_availability(profile)
        assert verdict.available is (auth_error is None)
        assert get_hermes_home_override() == str(home_a)
        assert current_secret_scope() is scope_a
        assert get_secret("PROFILE_KEY") == "from-a"
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def test_profile_worker_availability_refuses_stopped_gateway(tmp_path, monkeypatch):
    profile = ProfileInfo(
        name="stopped",
        path=tmp_path,
        is_default=False,
        gateway_running=False,
        provider="openai-codex",
        model="gpt-5.6-sol",
    )
    from hermes_cli import models

    monkeypatch.setattr(
        models,
        "_provider_has_credentials",
        lambda _provider: pytest.fail("auth must not mask a stopped gateway"),
    )

    unavailable = profile_availability.profile_worker_availability(profile)

    assert unavailable == profile_availability.ProfileWorkerAvailability(
        False,
        "profile 'stopped' gateway is stopped",
    )


def test_profile_worker_availability_restores_home_when_secret_scope_fails(
    tmp_path, monkeypatch,
):
    profile = ProfileInfo(
        name="broken-scope",
        path=tmp_path,
        is_default=False,
        gateway_running=True,
        provider="openai-codex",
        model="gpt-5.6-sol",
    )
    from agent import secret_scope
    from hermes_constants import get_hermes_home_override

    monkeypatch.setattr(
        secret_scope,
        "build_profile_secret_scope",
        lambda _path: (_ for _ in ()).throw(RuntimeError("scope unavailable")),
    )

    verdict = profile_availability.profile_worker_availability(profile)

    assert verdict.available is False
    assert "scope unavailable" in verdict.reason
    assert get_hermes_home_override() is None
