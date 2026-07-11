"""
Auth dependency — JWT-backed operator sessions.

`require_admin` now decodes the bearer JWT and returns the current `User`.
Backed by a real users table + bcrypt (see services/auth_service.py) — replaces
the old single shared password with in-memory tokens.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from database import get_db
from db_models import User
from services import auth_service
from logger import get_logger

logger = get_logger(__name__)

_bearer = HTTPBearer(auto_error=False)


def require_admin(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency: require a valid operator JWT; returns the User."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = auth_service.decode_token(credentials.credentials)
    user = auth_service.get_user(db, user_id) if user_id else None
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_portal(
    slug: str,
    user: User = Depends(require_admin),
) -> User:
    """Auth + tenant isolation for the per-client admin portal (/api/portal/{slug}).

    Authorizes a superadmin (sees any tenant) or the client_admin bound to THIS
    tenant. Anyone else gets a 404 so one client's staff can never see another's
    data — mirrors ``owned_client`` in api/clients.py.
    """
    if user.is_superadmin or user.client_slug == slug:
        return user
    raise HTTPException(status_code=404, detail=f"Portal '{slug}' not found")
