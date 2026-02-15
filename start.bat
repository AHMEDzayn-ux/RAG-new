@echo off
echo ========================================
echo Starting RAG Chatbot System
echo ========================================
echo.

echo [1/2] Starting Backend Server...
cd /d "%~dp0backend"
start "RAG Backend" cmd /k "venv\Scripts\activate && uvicorn main:app --host 0.0.0.0 --port 8000"

echo Waiting for backend to initialize...
timeout /t 10 /nobreak >nul

echo.
echo [2/2] Starting Frontend Server...
cd /d "%~dp0frontend"
start "RAG Frontend" cmd /k "npm run dev"

echo.
echo ========================================
echo System Starting...
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Press any key to open the application in your browser...
pause >nul

start http://localhost:3000

echo.
echo Both servers are running in separate windows.
echo Close those windows to stop the servers.
echo.
pause
