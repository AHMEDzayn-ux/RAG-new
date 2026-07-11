# Running Backend + Frontend Locally

## Backend (FastAPI, port 8000)

```powershell
cd D:\RAG\RAG-new\backend
.\venv\Scripts\python.exe main.py
```

No auto-reload — restart this process after any backend Python change.
Ready when you see `Application startup complete.`

## Frontend (Vite, port 3000)

```powershell
cd D:\RAG\RAG-new\frontend
npm run dev
```

Hot-reloads on save — no restart needed for frontend changes.
Ready when you see `Local: http://localhost:3000/`.

## Quick links once both are running

- Frontend: http://localhost:3000
- Backend API docs: http://localhost:8000/docs
- Operator console: http://localhost:3000/admin
- Client portal: http://localhost:3000/portal/{slug} (e.g. `/portal/nexus`)

## One-shot scripts

`start.ps1` / `start.bat` in the repo root launch both in separate windows
(uses `uvicorn main:app --host 0.0.0.0 --port 8000` instead of `python main.py`
directly, and opens the browser automatically).
