"""
Quick API validation test
"""
import requests
import json
import os

BASE_URL = "http://localhost:8000"

print("Testing Backend API...")
print("=" * 50)

# Test 1: Health
r = requests.get(f"{BASE_URL}/health")
print(f"✓ Health: {r.status_code}")

# Test 2: Create client
r = requests.post(f"{BASE_URL}/api/clients", json={
    "client_id": "demo",
    "collection_name": "demo_docs",
    "system_role": "helpful assistant"
})
print(f"✓ Create client: {r.status_code}")

# Test 3: Upload document  
with open("F:/My projects/RAG/documents/university_guide.pdf", 'rb') as f:
    r = requests.post(
        f"{BASE_URL}/api/clients/demo/documents",
        files=[('files', ('test.pdf', f, 'application/pdf'))],
        data={'category': 'test', 'doc_type': 'guide'}
    )
print(f"✓ Upload doc: {r.status_code} - {r.json().get('chunks_created', 0)} chunks")

# Test 4: Query with LLM
r = requests.post(f"{BASE_URL}/api/clients/demo/query", json={
    "question": "What is in this document?",
    "top_k": 2,
    "include_sources": True
})
print(f"✓ Query: {r.status_code}")
if r.status_code == 200:
    result = r.json()
    print(f"  Answer: {result['answer'][:100]}...")
    print(f"  Sources: {len(result.get('sources', []))}")
else:
    print(f"  Error: {r.json().get('detail', 'Unknown')[:100]}")

# Test 5: Chat
r = requests.post(f"{BASE_URL}/api/clients/demo/chat", json={
    "message": "Tell me more",
    "history": [],
    "use_retrieval": True
})
print(f"✓ Chat: {r.status_code}")
if r.status_code == 200:
    result = r.json()
    print(f"  Response: {result['response'][:100]}...")
else:
    print(f"  Error: {r.json().get('detail', 'Unknown')[:100]}")

# Cleanup
requests.delete(f"{BASE_URL}/api/clients/demo")
print(f"\n✓ All tests complete!")
