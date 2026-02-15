# 🤖 RAG Chatbot System - Complete Guide

## ✅ System Status

### Current Setup

- **Backend**: ✅ Running on http://localhost:8000
- **Frontend**: ⚠️ Vite server needs proper startup
- **Database**: ✅ SQLite at backend/rag_system.db
- **Documents Storage**: ✅ F:\My projects\RAG\documents
- **Vector Storage**: ✅ F:\My projects\RAG\vector_stores

---

## 📁 Directory Structure

```
F:\My projects\RAG\
├── backend/                      # FastAPI Backend
│   ├── api/                      # REST API endpoints
│   │   ├── clients.py           # Client management
│   │   ├── documents.py         # Document upload
│   │   ├── query.py             # Q&A and chat
│   │   └── models.py            # Pydantic schemas
│   ├── services/                # Core RAG logic
│   │   ├── document_loader.py   # PDF processing
│   │   ├── embeddings.py        # Vector embeddings
│   │   ├── llm_service.py       # Groq LLM integration
│   │   ├── vector_store.py      # FAISS vector DB
│   │   └── rag_pipeline.py      # Multi-tenant pipeline
│   ├── venv/                    # Python virtual environment
│   ├── .env                     # API keys & config
│   ├── config.py                # Settings management
│   ├── main.py                  # FastAPI app entry
│   └── requirements.txt         # Python dependencies
│
├── frontend/                    # React + Vite Frontend
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── ClientManager.jsx       # Client CRUD UI
│   │   │   ├── DocumentUpload.jsx      # File upload UI
│   │   │   └── ChatInterface.jsx       # Chat UI
│   │   ├── services/
│   │   │   └── api.js           # API client (axios)
│   │   ├── App.jsx              # Main app component
│   │   ├── main.jsx             # React entry point
│   │   └── index.css            # Global styles
│   ├── node_modules/            # NPM packages
│   ├── package.json             # NPM dependencies
│   ├── vite.config.js           # Vite configuration
│   └── index.html               # HTML template
│
├── documents/                   # 📄 PDF Storage
│   ├── sample_test.pdf         # Sample PDF files
│   └── university_guide.pdf
│
├── vector_stores/              # 🗄️ Vector Database Storage
│   └── faiss/                  # FAISS index files per client
│
├── check_system.py             # System status checker
└── test_quick.py               # API test script
```

---

## 🚀 How to Start the System

### Option 1: Manual Start (Recommended for Development)

**Terminal 1 - Backend:**

```powershell
cd "F:\My projects\RAG\backend"
.\venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**

```powershell
cd "F:\My projects\RAG\frontend"
npm run dev
```

### Option 2: Background Jobs

```powershell
# Start backend
$backend = Start-Job -ScriptBlock {
    Set-Location 'F:\My projects\RAG\backend'
    .\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
}

# Wait for backend startup
Start-Sleep -Seconds 8

# Start frontend (in foreground)
cd "F:\My projects\RAG\frontend"
npm run dev
```

### Option 3: Separate PowerShell Windows

```powershell
# Start backend in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'F:\My projects\RAG\backend'; .\venv\Scripts\Activate.ps1; uvicorn main:app --host 0.0.0.0 --port 8000"

# Start frontend in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'F:\My projects\RAG\frontend'; npm run dev"
```

---

## 🧪 Verify System is Working

Run the status checker:

```powershell
cd "F:\My projects\RAG"
python check_system.py
```

You should see all ✓ checkmarks.

---

## 🌐 Access Points

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

---

## 📋 How to Use the System

### 1. Create a Client

1. Open http://localhost:3000
2. Enter a client ID (e.g., "acme-corp")
3. Add description (optional)
4. Click "Create Client"
5. Click "Select" on your new client

### 2. Upload Documents

1. With client selected, scroll to "Document Management"
2. Click "Choose PDF file..."
3. Select a PDF from your computer
4. Click "Upload"
5. Wait for processing (shows chunk count)

### 3. Chat with Your Documents

1. Scroll to "Chat with Documents"
2. Choose mode:
   - **Chat Mode**: Conversational with context memory
   - **Query Mode**: One-shot Q&A without history
3. Type your question
4. Press Enter or click "Send"
5. View AI response with document sources

---

## 🗄️ Where Data is Stored

### PDF Documents

**Location**: `F:\My projects\RAG\documents\`

- Original PDF files uploaded by users
- Organized by client (in future multi-tenant setup)

### Vector Embeddings

**Location**: `F:\My projects\RAG\vector_stores\faiss\`

- FAISS index files for each client
- Format: `{client_id}.index`
- Example: `acme-corp.index`

### Client Metadata

**Location**: `F:\My projects\RAG\backend\rag_system.db`

- SQLite database (future feature)
- Stores client info, document metadata
- Currently managed in-memory

### Configuration

**Location**: `F:\My projects\RAG\backend\.env`

- API keys (Groq)
- LLM model settings
- System parameters

---

## 🔧 Configuration

### Backend Configuration (.env)

```env
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.7
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Frontend Configuration (vite.config.js)

```javascript
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    }
  }
}
```

---

## 🛠️ Troubleshooting

### Backend Won't Start

```powershell
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill process if needed
taskkill /PID <PID> /F

# Verify API key is set
cd backend
.\venv\Scripts\Activate.ps1
python -c "from config import settings; print(f'API Key: {settings.groq_api_key[:20]}...')"
```

### Frontend Won't Start

```powershell
# Check if port 3000 is in use
netstat -ano | findstr :3000

# Reinstall dependencies
cd frontend
Remove-Item -Recurse -Force node_modules
npm install
npm run dev
```

### Cannot Connect to Backend

1. Verify backend is running: http://localhost:8000/health
2. Check CORS settings in backend/main.py
3. Check firewall/antivirus blocking

### Upload Fails

1. Check documents directory exists: `F:\My projects\RAG\documents\`
2. Verify PDF file is valid (not corrupted)
3. Check backend logs for error details

### Chat Returns Empty Responses

1. Verify documents are uploaded
2. Check Groq API key is valid
3. View backend terminal for LLM errors

---

## 📊 API Endpoints

### Client Management

- `GET /api/clients` - List all clients
- `POST /api/clients` - Create new client
- `GET /api/clients/{id}` - Get client details
- `DELETE /api/clients/{id}` - Delete client

### Document Management

- `POST /api/clients/{id}/documents` - Upload PDF
- `GET /api/clients/{id}/documents` - List documents
- `DELETE /api/clients/{id}/documents` - Clear all documents

### Query & Chat

- `POST /api/clients/{id}/query` - Single Q&A
- `POST /api/clients/{id}/chat` - Conversational chat

---

## 🔍 Testing

### Manual API Test

```powershell
cd "F:\My projects\RAG"
python test_quick.py
```

### Test Individual Endpoints

```powershell
# Health check
curl http://localhost:8000/health

# List clients
curl http://localhost:8000/api/clients

# Create client
curl -X POST http://localhost:8000/api/clients -H "Content-Type: application/json" -d "{\"client_id\":\"test\",\"description\":\"Test client\"}"
```

---

## 🎯 Next Steps / Future Enhancements

1. **Authentication**: Add user login and JWT tokens
2. **Database**: Migrate from in-memory to persistent DB
3. **File Types**: Support DOC, DOCX, TXT beyond PDF
4. **UI Improvements**: Dark mode, file preview, etc.
5. **Deployment**: Docker containers for easy deployment
6. **Monitoring**: Logging, analytics, usage tracking
7. **Multi-user**: User accounts and permissions

---

## 📝 Notes

- Python 3.14 may show Pydantic warnings (safe to ignore)
- First document upload initializes embedding model (may take 10-30s)
- LLM responses depend on Groq API rate limits
- Vector store is FAISS (in-memory with file persistence)
- Frontend uses Vite HMR for instant updates during development

---

## 🆘 Support

If you encounter issues:

1. Run `python check_system.py` for diagnostics
2. Check backend terminal for errors
3. Check browser console (F12) for frontend errors
4. Verify all directories exist
5. Confirm API key is valid

---

**System Version**: Phase 8 Complete
**Last Updated**: February 5, 2026
