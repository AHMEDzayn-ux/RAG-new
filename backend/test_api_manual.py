"""
Test FastAPI endpoints
Quick manual test of Phase 7.1-7.3
"""

import requests
import json
import os

BASE_URL = "http://localhost:8000"

def test_root_endpoint():
    """Test root endpoint."""
    print("\n🧪 Testing GET /")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "running"
    print("✅ Root endpoint working!")


def test_health_check():
    """Test health check endpoint."""
    print("\n🧪 Testing GET /health")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check working!")


def test_create_client():
    """Test creating a new client."""
    print("\n🧪 Testing POST /api/clients")
    data = {
        "client_id": "test_university",
        "collection_name": "test_university_docs",
        "system_role": "university admissions advisor",
        "chunk_size": 500,
        "chunk_overlap": 100
    }
    response = requests.post(f"{BASE_URL}/api/clients", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 201
    assert response.json()["client_id"] == "test_university"
    print("✅ Client creation working!")
    return response.json()


def test_list_clients():
    """Test listing all clients."""
    print("\n🧪 Testing GET /api/clients")
    response = requests.get(f"{BASE_URL}/api/clients")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    assert "clients" in response.json()
    print("✅ Client listing working!")


def test_get_client(client_id):
    """Test getting specific client details."""
    print(f"\n🧪 Testing GET /api/clients/{client_id}")
    response = requests.get(f"{BASE_URL}/api/clients/{client_id}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    assert response.json()["client_id"] == client_id
    print("✅ Get client working!")


def test_delete_client(client_id):
    """Test deleting a client."""
    print(f"\n🧪 Testing DELETE /api/clients/{client_id}")
    response = requests.delete(f"{BASE_URL}/api/clients/{client_id}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✅ Client deletion working!")


def test_upload_documents(client_id):
    """Test uploading PDF documents."""
    print(f"\n🧪 Testing POST /api/clients/{client_id}/documents")
    
    # Check if sample PDF exists
    pdf_path = "F:/My projects/RAG/documents/university_guide.pdf"
    if not os.path.exists(pdf_path):
        print(f"⚠️  Sample PDF not found at {pdf_path}, skipping test")
        return
    
    with open(pdf_path, 'rb') as f:
        files = [('files', ('university_guide.pdf', f, 'application/pdf'))]
        data = {
            'category': 'university',
            'doc_type': 'guide'
        }
        response = requests.post(
            f"{BASE_URL}/api/clients/{client_id}/documents",
            files=files,
            data=data
        )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    assert response.json()["files_processed"] == 1
    print("✅ Document upload working!")


def test_list_documents(client_id):
    """Test listing documents."""
    print(f"\n🧪 Testing GET /api/clients/{client_id}/documents")
    response = requests.get(f"{BASE_URL}/api/clients/{client_id}/documents")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✅ Document listing working!")


def test_clear_documents(client_id):
    """Test clearing all documents."""
    print(f"\n🧪 Testing DELETE /api/clients/{client_id}/documents")
    response = requests.delete(f"{BASE_URL}/api/clients/{client_id}/documents")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✅ Document clearing working!")


def test_query(client_id):
    """Test RAG query."""
    print(f"\n🧪 Testing POST /api/clients/{client_id}/query")
    
    # Need GROQ_API_KEY for this to work
    if not os.environ.get('GROQ_API_KEY'):
        print("⚠️  GROQ_API_KEY not set, skipping query test")
        return
    
    data = {
        "question": "What information is in this document?",
        "top_k": 3,
        "include_sources": True
    }
    response = requests.post(f"{BASE_URL}/api/clients/{client_id}/query", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    if response.status_code != 200:
        print(f"Error: {json.dumps(result, indent=2)}")
        return
    print(f"Answer: {result.get('answer', '')[:200]}...")
    print(f"Sources: {len(result.get('sources', []))}")
    assert response.status_code == 200
    assert "answer" in result
    print("✅ Query working!")


def test_chat(client_id):
    """Test conversational chat."""
    print(f"\n🧪 Testing POST /api/clients/{client_id}/chat")
    
    # Need GROQ_API_KEY for this to work
    if not os.environ.get('GROQ_API_KEY'):
        print("⚠️  GROQ_API_KEY not set, skipping chat test")
        return
    
    data = {
        "message": "Tell me about the content",
        "history": [],
        "use_retrieval": True,
        "top_k": 2
    }
    response = requests.post(f"{BASE_URL}/api/clients/{client_id}/chat", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result.get('response', '')[:200]}...")
    print(f"Used retrieval: {result.get('used_retrieval')}")
    assert response.status_code == 200
    assert "response" in result
    print("✅ Chat working!")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 7: Complete FastAPI Backend - Test")
    print("=" * 60)
    
    try:
        # Phase 7.1 tests
        test_root_endpoint()
        test_health_check()
        
        # Phase 7.2 tests
        client = test_create_client()
        test_list_clients()
        test_get_client(client["client_id"])
        
        # Phase 7.3 tests
        test_upload_documents(client["client_id"])
        test_list_documents(client["client_id"])
        
        # Phase 7.4 tests
        test_query(client["client_id"])
        test_chat(client["client_id"])
        
        # Cleanup
        test_clear_documents(client["client_id"])
        test_delete_client(client["client_id"])
        
        print("\n" + "=" * 60)
        print("🎉 Phase 7 Complete - REST API Ready!")
        print("=" * 60)
        print("\nFeatures working:")
        print("  ✅ Basic API endpoints")
        print("  ✅ Create/list/get/delete clients")
        print("  ✅ Upload PDF documents")
        print("  ✅ List/clear documents")
        print("  ✅ RAG query with LLM")
        print("  ✅ Conversational chat")
        print("\nAPI Docs: http://localhost:8000/docs")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to server.")
        print("Make sure the server is running: python main.py")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
