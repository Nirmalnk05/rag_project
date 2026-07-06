"""Access context for per-user/role/site retrieval ACLs.

A request's principal is derived from headers (map your real auth/SSO to these).
When ACLs are disabled, or the principal is admin, everything is visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _csv(v: str | None) -> frozenset[str]:
    return frozenset(x.strip() for x in (v or "").split(",") if x.strip())


@dataclass(frozen=True)
class AccessContext:
    user_id: str = ""
    roles: frozenset[str] = field(default_factory=frozenset)
    sites: frozenset[str] = field(default_factory=frozenset)
    is_admin: bool = False

    @classmethod
    def admin(cls) -> "AccessContext":
        return cls(user_id="admin", is_admin=True)

    @classmethod
    def from_headers(cls, headers) -> "AccessContext":
        """Build from X-User-Id / X-Roles / X-Sites (comma-separated).
        `admin` in roles grants full visibility."""
        roles = _csv(headers.get("X-Roles"))
        return cls(
            user_id=headers.get("X-User-Id", ""),
            roles=roles,
            sites=_csv(headers.get("X-Sites")),
            is_admin="admin" in roles,
        )


def acl_visible(meta: dict, access: AccessContext | None) -> bool:
    """A chunk is visible if ACLs are off/admin, or the principal satisfies every
    ACL dimension the chunk restricts (role-gated AND site-gated independently).
    A chunk with no ACL metadata is public."""
    if access is None or access.is_admin:
        return True
    roles = meta.get("acl_roles") or []
    sites = meta.get("acl_sites") or []
    if roles and not (set(roles) & access.roles):
        return False
    if sites and not (set(sites) & access.sites):
        return False
    return True
