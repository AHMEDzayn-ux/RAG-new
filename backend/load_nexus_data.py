"""
Load Nexus Telecommunication Data with Semantic Metadata Enrichment

This script loads mobile packages and policies from JSON files into the RAG system
with intelligent metadata extraction and semantic enrichment for optimal retrieval.

Usage:
    python backend/load_nexus_data.py
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from services.document_loader import DocumentLoader
from services.embeddings import EmbeddingsService
from services.vector_store_faiss import VectorStoreService
from logger import get_logger
import json

logger = get_logger(__name__)


def load_nexus_telecommunications():
    """Load Nexus Telecom packages and policies with semantic metadata."""
    
    # Initialize services
    logger.info("Initializing RAG services...")
    doc_loader = DocumentLoader()
    embedding_service = EmbeddingsService()
    vector_store = VectorStoreService()  # No embedding_service parameter needed
    
    # Paths
    source_dir = Path(__file__).parent.parent / "source files"
    packages_file = source_dir / "packages.json"
    policies_file = source_dir / "policies.json"
    
    # Collection name (matches frontend client_id with "client_" prefix for WhatsApp)
    collection_name = "client_Nexus"
    
    logger.info(f"Loading Nexus Telecommunication data into collection: {collection_name}")
    
    # ========================================
    # Load Mobile Packages
    # ========================================
    logger.info(f"\n{'='*60}")
    logger.info("LOADING MOBILE PACKAGES")
    logger.info(f"{'='*60}")
    
    if not packages_file.exists():
        logger.error(f"Packages file not found: {packages_file}")
        return
    
    logger.info(f"Reading: {packages_file}")
    with open(packages_file, 'r', encoding='utf-8') as f:
        packages_data = json.load(f)
    
    # Field mapping for packages
    text_fields = [
        "name",           # Package name (main text)
        "category",       # Category description
    ]
    
    metadata_fields = [
        "package_id",
        "category",
        "validity_days",
        "price_lkr",
        "tags",
        "benefits",        # Nested dict - will be used for semantic enrichment
        "activation",      # Nested dict
        "eligibility",     # Nested dict
        "auto_renew",
        "stackable",
        "fair_usage_policy",
        "regions_supported"  # For roaming packages
    ]
    
    logger.info(f"Processing {len(packages_data['packages'])} packages...")
    logger.info(f"Text fields: {text_fields}")
    logger.info(f"Metadata fields: {metadata_fields}")
    
    # Chunk packages with semantic enrichment
    package_chunks = doc_loader.load_and_chunk_json(
        file_path=str(packages_file),
        array_field="packages",
        text_fields=text_fields,
        metadata_fields=metadata_fields,
        group_size=1,  # One package per chunk for precise retrieval
        metadata={
            "source_file": "packages.json",
            "document_type": "mobile_package",
            "company": "Nexus Telecommunication"
        }
    )
    
    logger.info(f"✅ Created {len(package_chunks)} package chunks with semantic metadata")
    
    # Show example of enriched chunk
    if package_chunks:
        logger.info(f"\n{'='*60}")
        logger.info("EXAMPLE ENRICHED PACKAGE CHUNK:")
        logger.info(f"{'='*60}")
        example = package_chunks[0]
        logger.info(f"Package ID: {example['metadata'].get('package_id', 'N/A')}")
        logger.info(f"Category: {example['metadata'].get('category', 'N/A')}")
        logger.info(f"Price: LKR {example['metadata'].get('price_lkr', 'N/A')}")
        logger.info(f"\nEnriched Text Preview (first 500 chars):")
        logger.info("-" * 60)
        logger.info(example['text'][:500] + "...")
        logger.info("-" * 60)
    
    # ========================================
    # Load Policies
    # ========================================
    logger.info(f"\n{'='*60}")
    logger.info("LOADING COMPANY POLICIES")
    logger.info(f"{'='*60}")
    
    if not policies_file.exists():
        logger.error(f"Policies file not found: {policies_file}")
        return
    
    logger.info(f"Reading: {policies_file}")
    with open(policies_file, 'r', encoding='utf-8') as f:
        policies_data = json.load(f)
    
    # Field mapping for policies
    policy_text_fields = [
        "title",
        "rules"
    ]
    
    policy_metadata_fields = [
        "policy_id",
        "title",
        "applies_to",
        "violation_consequences"
    ]
    
    logger.info(f"Processing {len(policies_data['policies'])} policies...")
    logger.info(f"Text fields: {policy_text_fields}")
    logger.info(f"Metadata fields: {policy_metadata_fields}")
    
    # Chunk policies
    policy_chunks = doc_loader.load_and_chunk_json(
        file_path=str(policies_file),
        array_field="policies",
        text_fields=policy_text_fields,
        metadata_fields=policy_metadata_fields,
        group_size=1,  # One policy per chunk
        metadata={
            "source_file": "policies.json",
            "document_type": "company_policy",
            "company": "Nexus Telecommunication"
        }
    )
    
    logger.info(f"✅ Created {len(policy_chunks)} policy chunks")
    
    # ========================================
    # Combine and Store in Vector Store
    # ========================================
    logger.info(f"\n{'='*60}")
    logger.info("STORING IN VECTOR DATABASE")
    logger.info(f"{'='*60}")
    
    all_chunks = package_chunks + policy_chunks
    logger.info(f"Total chunks to embed: {len(all_chunks)}")
    
    # Extract texts and metadata
    texts = [chunk['text'] for chunk in all_chunks]
    metadatas = [chunk['metadata'] for chunk in all_chunks]
    ids = [f"{collection_name}_{i}" for i in range(len(all_chunks))]
    
    # Create collection
    logger.info(f"Creating collection: {collection_name}")
    vector_store.create_collection(name=collection_name)
    
    # Generate embeddings
    logger.info("Generating embeddings and storing documents...")
    embeddings = embedding_service.embed_batch(texts, show_progress=True)
    
    # Add documents with embeddings
    vector_store.add_documents(
        collection_name=collection_name,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    # Persist collection to disk
    logger.info("Persisting collection to disk...")
    vector_store.persist()
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ SUCCESS - NEXUS TELECOM DATA LOADED")
    logger.info(f"{'='*60}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Total documents: {len(all_chunks)}")
    logger.info(f"  - Packages: {len(package_chunks)}")
    logger.info(f"  - Policies: {len(policy_chunks)}")
    logger.info(f"\nSemantic Features Enabled:")
    logger.info("  ✅ Metadata embedded into searchable content")
    logger.info("  ✅ Synonym expansion for categories and tags")
    logger.info("  ✅ Nested benefits extraction (data, voice, SMS)")
    logger.info("  ✅ Price tier semantic tags (budget/premium)")
    logger.info("  ✅ Validity period variations (daily/weekly/monthly)")
    logger.info(f"\nExample Queries:")
    logger.info("  • 'I need a cheap data plan for daily use'")
    logger.info("  • 'What unlimited social media packages do you have?'")
    logger.info("  • 'Show me gaming plans with low latency'")
    logger.info("  • 'Do you have international roaming for Asia?'")
    logger.info("  • 'What's the refund policy?'")
    logger.info(f"\nMetadata Filtering Examples:")
    logger.info("  • metadata_filter={'category': 'data'}")
    logger.info("  • metadata_filter={'tags': 'budget'}")
    logger.info("  • metadata_filter={'category': 'roaming', 'validity_days': 7}")
    logger.info(f"\n{'='*60}\n")
    
    return collection_name


def test_semantic_search():
    """Test semantic search capabilities."""
    logger.info(f"\n{'='*60}")
    logger.info("TESTING SEMANTIC SEARCH")
    logger.info(f"{'='*60}\n")
    
    # Initialize services
    embedding_service = EmbeddingsService()
    vector_store = VectorStoreService()  # No embedding_service parameter needed
    
    collection_name = "client_Nexus"
    
    # Load collection from disk
    logger.info(f"Loading collection '{collection_name}' from disk...")
    try:
        vector_store.load_collection(collection_name)
        doc_count = vector_store.get_collection_count(collection_name)
        logger.info(f"Successfully loaded collection with {doc_count} documents")
    except Exception as e:
        logger.error(f"Failed to load collection: {e}")
        return
    
    # Test queries
    test_queries = [
        "I need a cheap data plan for daily use",
        "What unlimited social media packages do you have?",
        "Show me gaming plans with low latency",
        "Do you have international roaming for Asia?",
        "What's your refund policy?",
        "I'm a student, what plans are for me?",
        "Affordable monthly internet package"
    ]
    
    for query in test_queries:
        logger.info(f"\n🔍 Query: '{query}'")
        logger.info("-" * 60)
        
        # Get embeddings
        query_embedding = embedding_service.embed_batch([query])
        
        # Search
        results = vector_store.query(
            collection_name=collection_name,
            query_embeddings=query_embedding,
            n_results=3
        )
        
        if results and results['documents']:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ), 1):
                logger.info(f"\n  Result {i} (Distance: {distance:.4f}):")
                logger.info(f"  Package: {metadata.get('name', metadata.get('title', 'N/A'))}")
                logger.info(f"  Category: {metadata.get('category', 'N/A')}")
                logger.info(f"  Price: LKR {metadata.get('price_lkr', 'N/A')}")
                if metadata.get('tags'):
                    logger.info(f"  Tags: {', '.join(metadata['tags']) if isinstance(metadata['tags'], list) else metadata['tags']}")
        else:
            logger.info("  No results found")
    
    logger.info(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load Nexus Telecommunication data")
    parser.add_argument("--test", action="store_true", help="Run semantic search tests after loading")
    args = parser.parse_args()
    
    try:
        # Load data
        collection_name = load_nexus_telecommunications()
        
        # Test if requested
        if args.test and collection_name:
            test_semantic_search()
        
        logger.info("🎉 All done! You can now query the Nexus Telecom collection via the API.")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)
