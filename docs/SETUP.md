# Python Installation & Setup Guide

## ⚠️ Python Not Found

Python needs to be installed before continuing with the project setup.

## 📥 Install Python

### Option 1: Official Python Installer (Recommended)

1. Visit: https://www.python.org/downloads/
2. Download Python 3.11 or 3.12 (recommended for this project)
3. **IMPORTANT**: During installation, check "Add Python to PATH"
4. Click "Install Now"
5. Restart your terminal/PowerShell

### Option 2: Using Winget (Windows Package Manager)

```powershell
winget install Python.Python.3.12
```

### Option 3: Using Chocolatey

```powershell
choco install python312
```

## ✅ Verify Installation

After installing, open a NEW PowerShell window and run:

```powershell
python --version
# Should output: Python 3.11.x or 3.12.x

pip --version
# Should output pip version
```

## 🚀 Continue Setup After Python Installation

Once Python is installed, run these commands:

```powershell
# Navigate to backend folder
cd "F:\My projects\RAG\backend"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies
pip install -r requirements.txt

# Verify installation
pytest --version
```

## 🔑 Get Groq API Key (Free)

1. Visit: https://console.groq.com/
2. Sign up with email/Google
3. Navigate to API Keys section
4. Click "Create API Key"
5. Copy the key

## 📝 Configure Environment

```powershell
# Copy example env file
cp .env.example .env

# Edit .env file and add your Groq API key
notepad .env
```

Replace `your_groq_api_key_here` with your actual API key.

## ✅ Test Setup

```powershell
# Run tests to verify everything works
pytest

# Expected output: tests collected, environment ready
```

## 📞 Need Help?

If you encounter any issues:

- Ensure Python version is 3.11 or 3.12
- Make sure PATH is configured correctly
- Try restarting your terminal after installation
- Run PowerShell as Administrator if permission errors occur

---

**Next Steps**: Once Python is installed and dependencies are ready, we'll move to Phase 2: PDF loading & chunking.
