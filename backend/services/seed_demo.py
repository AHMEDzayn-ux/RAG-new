"""
Demo seeding — makes a fresh/wiped deployment self-heal.

On free/ephemeral hosts the disk can be wiped on deploy/cold-start/restart, so
demo clients (nexus, unihelp), their FAISS knowledge-base collections, and the
operator portal dataset can disappear. Observed in production: the SQLite DB
(client rows) survived a restart while the FAISS vector-store directory did
NOT — so the client existed but `search_knowledge_base` hit "Collection
'client_nexus' does not exist" and the bot honestly had nothing to answer
from, even though documents/nexus_knowledge_base.json was right there. This
module repairs each piece independently — client row, KB collection, operator
dataset — on every startup, so a partial wipe self-heals too.

Idempotent: only ever fills in what's missing, so it's a no-op on a fully
durable deployment and never clobbers real data. Defensive: one client's
failure won't break app startup.
"""

from sqlalchemy.orm import Session

from services import client_store, telecom_store
from telecom_models import Customer
from db_models import Document
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
            is_new = client is None
            if is_new:
                client = client_store.create_client(
                    db, slug=slug, name=spec["name"], domain=spec["domain"],
                )  # owner_id NULL -> claimed by the bootstrap admin right after

            # Repair the KB vector-store collection whenever it's missing or empty —
            # this covers a brand-new client AND (the bug actually seen in prod) a
            # pre-existing client whose FAISS collection got wiped independently of
            # its DB row. get_pipeline() lazy-loads from disk first so a healthy,
            # already-persisted collection is never needlessly re-indexed.
            manager = get_pipeline_manager()
            pipeline = manager.get_pipeline(slug) or manager.create_pipeline(
                client_id=slug,
                system_role=client_store.resolve_persona(client),
                domain=spec["domain"],
            )
            try:
                has_kb = pipeline.vector_store.get_collection_count(pipeline.collection_name) > 0
            except ValueError:
                has_kb = False

            if not has_kb:
                kb_path = settings.documents_dir / spec["kb"]
                if kb_path.exists():
                    result = pipeline.index_documents(
                        file_paths=[str(kb_path)],
                        metadata={"category": "seed", "doc_type": "knowledge_base"},
                    )
                    chunks = result.get("chunks_created", result.get("total_chunks", 0)) \
                        if isinstance(result, dict) else 0
                    already_logged = db.query(Document.id).filter(
                        Document.client_slug == slug, Document.filename == spec["kb"],
                    ).first() is not None
                    if not already_logged:
                        try:
                            client_store.add_document(
                                db, client_slug=slug, filename=spec["kb"],
                                doc_type="json", chunk_count=chunks,
                            )
                        except Exception:
                            pass
                    logger.info(
                        f"{'Seeded' if is_new else 'Repaired'} demo client '{slug}' KB with {chunks} chunks"
                    )
                else:
                    logger.warning(f"Seed KB not found for '{slug}': {kb_path}")

            if is_new:
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
