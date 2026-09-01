"""Tenant resolution and authorization guards.

Authentication is deliberately kept behind a thin abstraction so it can be
replaced by OAuth/SSO without touching the domain code: every request resolves
to a :class:`TenantContext`, and every query in the application is scoped by
``ctx.tenant_id``.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db import get_db

DEMO_TENANT_SLUG = "demo"


@dataclass(frozen=True)
class TenantContext:
    tenant_id: uuid.UUID
    tenant_slug: str
    user_id: uuid.UUID | None
    role: str

    def require(self, *roles: str) -> None:
        if roles and self.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"role '{self.role}' is not allowed to perform this action",
            )


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    return "pk_" + secrets.token_urlsafe(32)


def verify_api_key(raw_key: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_api_key(raw_key), stored_hash)


def current_tenant(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> TenantContext:
    from app.models import Tenant  # local import keeps this module import-light

    if x_api_key:
        tenant = db.scalar(
            select(Tenant).where(Tenant.api_key_hash == hash_api_key(x_api_key))
        )
        if tenant is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key"
            )
        return TenantContext(tenant.id, tenant.slug, None, "admin")

    if not settings.allow_anonymous_demo_tenant:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="X-API-Key header required"
        )

    tenant = db.scalar(select(Tenant).where(Tenant.slug == DEMO_TENANT_SLUG))
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="no demo tenant available; provide an X-API-Key header",
        )
    return TenantContext(tenant.id, tenant.slug, None, "admin")
