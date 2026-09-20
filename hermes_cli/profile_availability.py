"""Read-only worker readiness checks for Hermes profiles.

Kanban routing must not treat an installed profile as runnable merely because
its directory exists.  A worker needs a live gateway, a configured model and
provider, and credentials in that profile's isolated secret scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from hermes_cli.profiles import ProfileInfo


@dataclass(frozen=True)
class ProfileWorkerAvailability:
    """Whether a profile can accept a Kanban worker right now."""

    available: bool
    reason: str = ""


def profile_worker_availability(
    profile: "ProfileInfo",
    *,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> ProfileWorkerAvailability:
    """Measure whether ``profile`` can start a worker, without mutating auth.

    The home and secret scopes are bound explicitly because a multiplexed
    gateway serves several isolated profiles in one process.  Provider status
    helpers are read-only; OAuth checks do not adopt or refresh credentials.
    """
    name = str(getattr(profile, "name", "") or "").strip() or "(unknown)"
    if not bool(getattr(profile, "gateway_running", False)):
        return ProfileWorkerAvailability(False, f"profile {name!r} gateway is stopped")

    provider = str(provider_override or getattr(profile, "provider", "") or "").strip()
    model = str(model_override or getattr(profile, "model", "") or "").strip()
    if not provider:
        return ProfileWorkerAvailability(False, f"profile {name!r} has no provider configured")
    if not model:
        return ProfileWorkerAvailability(False, f"profile {name!r} has no model configured")

    try:
        from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.models import _provider_has_credentials

        profile_home = Path(getattr(profile, "path"))
        home_token = set_hermes_home_override(profile_home)
        secret_token = None
        try:
            secret_token = set_secret_scope(build_profile_secret_scope(profile_home))
            authenticated = _provider_has_credentials(provider)
        finally:
            if secret_token is not None:
                reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)
    except Exception as exc:
        return ProfileWorkerAvailability(
            False,
            f"profile {name!r} readiness check failed: {type(exc).__name__}: {exc}",
        )

    if not authenticated:
        return ProfileWorkerAvailability(
            False,
            f"profile {name!r} has no usable credentials for provider {provider!r}",
        )
    return ProfileWorkerAvailability(True)


def worker_availability_for_name(
    profile_name: str,
    *,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> ProfileWorkerAvailability:
    """Resolve ``profile_name`` and measure its worker readiness."""
    try:
        from hermes_cli.profiles import list_profiles, normalize_profile_name

        canon = normalize_profile_name(profile_name)
        profile = next(
            (item for item in list_profiles(lazy_skill_count=True) if item.name == canon),
            None,
        )
    except Exception as exc:
        return ProfileWorkerAvailability(
            False,
            f"profile {profile_name!r} readiness check failed: {type(exc).__name__}: {exc}",
        )
    if profile is None:
        return ProfileWorkerAvailability(False, f"profile {canon!r} does not exist")
    return profile_worker_availability(
        profile,
        provider_override=provider_override,
        model_override=model_override,
    )
