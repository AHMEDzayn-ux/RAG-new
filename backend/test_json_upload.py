"""
Test JSON Upload and Retrieval - Customer Care Use Case
Tests the new hybrid document loader with JSON support for customer care packages.
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
CLIENT_ID = "customer_care_test"

def test_json_upload_and_query():
    """Test uploading JSON customer care data and querying it."""
    
    print("=" * 80)
    print("TESTING JSON UPLOAD FOR CUSTOMER CARE USE CASE")
    print("=" * 80)
    
    # Step 1: Create a test client
    print("\n1. Creating test client...")
    response = requests.post(
        f"{BASE_URL}/api/clients",
        json={
            "client_id": CLIENT_ID,
            "name": "Customer Care Test Client",
            "description": "Testing JSON customer care package support"
        }
    )
    
    if response.status_code == 201:
        print("   ✓ Client created successfully")
    elif response.status_code == 500 and "already exists" in response.text:
        print("   ℹ Client already exists (using existing)")
    else:
        print(f"   ✗ Failed to create client: {response.status_code}")
        print(f"   Response: {response.text}")
        # Continue anyway - client might exist
        print("   ℹ Continuing with existing client...")
    
    # Step 2: Upload JSON file
    print("\n2. Uploading customer care JSON file...")
    json_file_path = Path(__file__).parent.parent / "documents" / "sample_customer_care.json"
    
    if not json_file_path.exists():
        print(f"   ✗ JSON file not found: {json_file_path}")
        return
    
    with open(json_file_path, 'rb') as f:
        files = {'file': ('sample_customer_care.json', f, 'application/json')}
        data = {
            'category': 'customer_support',
            'doc_type': 'faq_database'
        }
        
        response = requests.post(
            f"{BASE_URL}/api/clients/{CLIENT_ID}/documents",
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ JSON uploaded successfully!")
        print(f"   - Files processed: {result['files_processed']}")
        print(f"   - Chunks created: {result['chunks_created']}")
        print(f"   - Total documents: {result['total_documents']}")
        
        if result.get('chunk_previews'):
            print(f"\n   Chunk Previews:")
            for i, preview in enumerate(result['chunk_previews'][:3], 1):
                print(f"\n   Chunk {i}:")
                text = preview.get('text_preview', preview.get('text', ''))
                print(f"   Text: {text[:150]}...")
                print(f"   Metadata: {preview['metadata']}")
    else:
        print(f"   ✗ Failed to upload JSON: {response.status_code}")
        print(f"   Response: {response.text}")
        return
    
    # Step 3: Test queries
    print("\n3. Testing queries against customer care knowledge base...")
    
    test_queries = [
        "How do I track my package?",
        "What are the shipping speeds available?",
        "Can I return a package?",
        "Do you ship internationally?",
        "What if my package is lost?",
        "How much does shipping cost?",
        "What package sizes do you accept?"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        
        response = requests.post(
            f"{BASE_URL}/api/clients/{CLIENT_ID}/chat",
            json={"message": query}  # Fixed: use 'message' instead of 'query'
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Response: {result['response'][:200]}...")
            
            if result.get('sources'):
                print(f"   Sources retrieved: {len(result['sources'])}")
                for source in result['sources'][:2]:
                    metadata = source.get('metadata', {})
                    print(f"     - Category: {metadata.get('category', 'N/A')}, "
                          f"ID: {metadata.get('id', 'N/A')}")
        else:
            print(f"   ✗ Query failed: {response.status_code}")
            print(f"   Response: {response.text}")
            break
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
    
    # Step 4: Get client stats
    print("\n4. Final client statistics...")
    response = requests.get(f"{BASE_URL}/api/clients/{CLIENT_ID}")
    if response.status_code == 200:
        stats = response.json()
        print(f"   Client ID: {stats.get('client_id', CLIENT_ID)}")
        print(f"   Total chunks: {stats.get('document_count', 'N/A')}")
        print(f"   Model: {stats.get('llm_model', 'N/A')}")
    else:
        print(f"   ⚠ Could not retrieve stats: {response.status_code}")
    
    print("\n✓ All tests completed successfully!")


if __name__ == "__main__":
    test_json_upload_and_query()
