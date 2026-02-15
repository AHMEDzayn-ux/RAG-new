@echo off
cd /d "F:\My projects\RAG\backend"
set GROQ_API_KEY=your_groq_api_key_here
call venv\Scripts\activate.bat
python main.py
