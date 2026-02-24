"""Quick test to see what text is actually stored for FUP_VIDEO policy"""
import sys
sys.path.insert(0, '.')

from services.vector_store import VectorStoreService
from services.embeddings import EmbeddingsService

# Initialize services
vector_store = VectorStoreService()
embeddings_service = EmbeddingsService()

# Load collection
vector_store.load_collection("client_Nexus")

# Generate embedding for query
query_embedding = embeddings_service.embed_text("video streaming fair usage")

# Query for FUP_VIDEO
results = vector_store.query(
    collection_name="client_Nexus",
    query_embeddings=[query_embedding],
    n_results=3
)

# Write to file to avoid Unicode issues
output_file = "policy_test_results.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("\n" + "="*80 + "\n")
    f.write("SEARCH RESULTS FOR: 'video streaming fair usage'\n")
    f.write("="*80 + "\n")

    if results and results['documents']:
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            f.write(f"\n[Document {i+1}]\n")
            f.write(f"Distance: {distance:.3f}\n")
            f.write(f"Policy ID: {metadata.get('policy_id', 'N/A')}\n")
            f.write(f"Title: {metadata.get('title', 'N/A')}\n")
            f.write(f"\nStored Text:\n")
            f.write("-" * 80 + "\n")
            f.write(doc + "\n")
            f.write("-" * 80 + "\n")
            f.write(f"\nOriginal Text (if different):\n")
            if 'original_text' in metadata:
                f.write(metadata['original_text'] + "\n")
            f.write("\n" + "="*80 + "\n")

print(f"Results written to: {output_file}")
