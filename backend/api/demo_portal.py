"""Interactive Nexus demo backed by isolated, disposable SQLite sessions."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from secrets import token_urlsafe
from threading import Lock
from time import monotonic

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from api.models import LoginRequest
from api.portal import router as portal_router
from auth import require_portal
from database import Base, get_db
from db_models import Client
from services.telecom_store import seed_telecom_demo

DEMO_SLUG = "nexus"
# These public form values grant access only to disposable sample data.
DEMO_EMAIL = "demo-admin@nexus.local"
DEMO_PASSWORD = "NexusDemo123"
SESSION_IDLE_SECONDS = 30 * 60
MAX_SESSIONS = 24
MAX_MUTATIONS = 100
MAX_MUTATION_BYTES = 16 * 1024


@dataclass
class DemoSession:
    engine: Engine
    last_used: float = field(default_factory=monotonic)
    lock: Lock = field(default_factory=Lock)
    in_use: int = 0
    mutations: int = 0


_sessions: dict[str, DemoSession] = {}
_sessions_lock = Lock()
_bearer = HTTPBearer(auto_error=False)


def _prune_sessions(now: float) -> None:
    expired = [
        token for token, demo in _sessions.items()
        if not demo.in_use and now - demo.last_used >= SESSION_IDLE_SECONDS
    ]
    for token in expired:
        _sessions.pop(token).engine.dispose()


def _new_demo_engine() -> Engine:
    # Never use the configured application engine or copy the live database.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    try:
        Base.metadata.create_all(engine)
        with Session(engine, autoflush=False) as db:
            db.add(Client(slug=DEMO_SLUG, name="Nexus Telecom", domain="telecom"))
            db.commit()
            seed_telecom_demo(db, DEMO_SLUG)
    except Exception:
        engine.dispose()
        raise
    return engine


demo_app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)


@demo_app.post("/session", status_code=201)
def open_demo_session(login: LoginRequest, response: Response) -> dict[str, str]:
    if login.email.strip().lower() != DEMO_EMAIL or login.password != DEMO_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid demo login.")
    with _sessions_lock:
        _prune_sessions(monotonic())
        if len(_sessions) >= MAX_SESSIONS:
            idle = [(demo.last_used, token) for token, demo in _sessions.items() if not demo.in_use]
            if not idle:
                raise HTTPException(status_code=503, detail="The demo is busy. Please try again.")
            _, oldest = min(idle)
            _sessions.pop(oldest).engine.dispose()
        token = token_urlsafe(32)
        _sessions[token] = DemoSession(_new_demo_engine())

    response.headers["Cache-Control"] = "no-store"
    return {"token": token}


@contextmanager
def _lease_demo(token: str):
    with _sessions_lock:
        _prune_sessions(monotonic())
        demo = _sessions.get(token)
        if demo is None:
            raise HTTPException(status_code=401, detail="Demo session ended. Start a new demo.")
        demo.in_use += 1
    try:
        yield demo
    finally:
        with _sessions_lock:
            demo.in_use -= 1
            demo.last_used = monotonic()


async def limit_demo_body(request: Request) -> None:
    if request.method in {"POST", "PATCH"} and len(await request.body()) > MAX_MUTATION_BYTES:
        raise HTTPException(status_code=413, detail="This demo request is too large.")


def demo_db(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
):
    if credentials is None:
        raise HTTPException(status_code=401, detail="Start a demo session first.")
    with _lease_demo(credentials.credentials) as demo, demo.lock:
        if request.method in {"POST", "PATCH"}:
            if demo.mutations >= MAX_MUTATIONS:
                raise HTTPException(status_code=429, detail="Demo edit limit reached. Start a new demo.")
            demo.mutations += 1
        with Session(demo.engine, autoflush=False) as db:
            yield db


def require_demo_portal(slug: str) -> None:
    if slug != DEMO_SLUG:
        raise HTTPException(status_code=404, detail="Demo portal not found.")


# These overrides apply only to the mounted demo app.
demo_app.dependency_overrides[get_db] = demo_db
demo_app.dependency_overrides[require_portal] = require_demo_portal
demo_app.include_router(portal_router, dependencies=[Depends(limit_demo_body)])
