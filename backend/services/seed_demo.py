"""
Demo seeding — makes a fresh/wiped deployment self-heal.

On free hosts (e.g. Render free tier) the disk is ephemeral: the SQLite DB and
FAISS indexes are wiped on every deploy/cold-start, so demo clients (nexus,
unihelp) and their knowledge bases disappear and /c/{slug} 404s. This recreates
them idempotently on startup so the public demo always works.

Idempotent: any client that already exists is skipped, so it's a no-op on a
durable DB and never clobbers real data. Defensive: one client's failure won't
break app startup.
"""

from sqlalchemy.orm import Session

from services import client_store, telecom_store
from telecom_models import Customer
from config import get_settings
from logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Demo tenants restored on a fresh deploy. KB files live in documents/.
DEMO_CLIENTS = [
    {"slug": "nexus", "name": "Nexus Telecom", "domain": "telecom",
     "kb": "nexus_knowledge_base.json"},
    {"slug": "unihelp", "name": "UniHelp", "domain": "university",
     "kb": "university_knowledge_base.json"},
]


def seed_demo_clients(db: Session) -> int:
    """Create any missing demo client (+ its KB + mock accounts). Returns count seeded."""
    if not settings.seed_demo_on_startup:
        return 0
    # Imported here to avoid a circular import at module load.
    from api.clients import get_pipeline_manager

    seeded = 0
    for spec in DEMO_CLIENTS:
        slug = spec["slug"]
        try:
            client = client_store.get_client(db, slug)
            if client is None:
                client = client_store.create_client(
                    db, slug=slug, name=spec["name"], domain=spec["domain"],
                )  # owner_id NULL -> claimed by the bootstrap admin right after

                manager = get_pipeline_manager()
                pipeline = manager.create_pipeline(
                    client_id=slug,
                    system_role=client_store.resolve_persona(client),
                    domain=spec["domain"],
                )

                kb_path = settings.documents_dir / spec["kb"]
                if kb_path.exists():
                    result = pipeline.index_documents(
                        file_paths=[str(kb_path)],
                        metadata={"category": "seed", "doc_type": "knowledge_base"},
                    )
                    chunks = result.get("chunks_created", result.get("total_chunks", 0)) \
                        if isinstance(result, dict) else 0
                    try:
                        client_store.add_document(
                            db, client_slug=slug, filename=spec["kb"],
                            doc_type="json", chunk_count=chunks,
                        )
                    except Exception:
                        pass
                    logger.info(f"Seeded demo client '{slug}' with {chunks} KB chunks")
                else:
                    logger.warning(f"Seed KB not found for '{slug}': {kb_path}")

                # Demo mock accounts so account lookup/change works out of the box.
                try:
                    client_store.seed_demo_accounts(db, slug, spec["domain"])
                except Exception as e:
                    logger.warning(f"Could not seed accounts for '{slug}': {e}")

                seeded += 1

            # Operator portal dataset (customers/subscriptions/CDRs/tickets etc.) used
            # to require clicking "Load demo data" by hand. Auto-seed it whenever it's
            # missing — including on a client that already existed (e.g. the client
            # row survived but this tenant's telecom rows are empty) — never on a
            # tenant that already has real data, so this never clobbers anything.
            if spec["domain"] == "telecom":
                has_portal_data = db.query(Customer.id).filter(
                    Customer.client_slug == slug
                ).first() is not None
                if not has_portal_data:
                    telecom_store.seed_telecom_demo(db, slug)
                    logger.info(f"Seeded telecom operator demo dataset for '{slug}'")
        except Exception as e:
            logger.error(f"Failed to seed demo client '{slug}': {e}")
    return seeded
