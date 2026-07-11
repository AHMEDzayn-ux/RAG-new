"""
System Status Checker for RAG Chatbot
Verifies all components are working correctly
"""

import os
import sys
from pathlib import Path
import requests

def check_directories():
    """Check all required directories exist"""
    print("📁 Checking Directories...")
    
    base = Path(__file__).parent
    dirs = {
        'Backend': base / 'backend',
        'Frontend': base / 'frontend',
        'Documents': base / 'documents',
        'Vector Stores': base / 'vector_stores',
        'Frontend/src': base / 'frontend' / 'src',
        'Frontend/src/components': base / 'frontend' / 'src' / 'components',
        'Frontend/src/services': base / 'frontend' / 'src' / 'services',
    }
    
    all_ok = True
    for name, path in dirs.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_ok = False
    
    return all_ok

def check_backend():
    """Check backend API is responding"""
    print("\n🔧 Checking Backend API...")
    
    try:
        # Health check
        r = requests.get('http://localhost:8000/health', timeout=5)
        if r.status_code == 200:
            print(f"  ✓ Health endpoint: {r.json()}")
        else:
            print(f"  ✗ Health endpoint returned {r.status_code}")
            return False
        
        # List clients
        r = requests.get('http://localhost:8000/api/clients', timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"  ✓ Clients endpoint: {len(data.get('clients', []))} clients")
        else:
            print(f"  ✗ Clients endpoint returned {r.status_code}")
            return False
        
        return True
    except requests.exceptions.ConnectionError:
        print("  ✗ Cannot connect to backend. Is it running on http://localhost:8000?")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def check_frontend():
    """Check frontend is serving"""
    print("\n🌐 Checking Frontend...")
    
    try:
        r = requests.get('http://localhost:3000', timeout=5)
        if r.status_code == 200:
            print(f"  ✓ Frontend is serving (Status {r.status_code})")
            return True
        else:
            print(f"  ✗ Frontend returned {r.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("  ✗ Cannot connect to frontend. Is it running on http://localhost:3000?")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def check_config():
    """Check configuration"""
    print("\n⚙️ Checking Configuration...")
    
    env_file = Path(__file__).parent / 'backend' / '.env'
    if env_file.exists():
        print(f"  ✓ .env file exists")
        with open(env_file) as f:
            content = f.read()
            if 'GROQ_API_KEY=' in content and len(content.split('GROQ_API_KEY=')[1].split('\n')[0].strip()) > 10:
                print(f"  ✓ Groq API key configured")
            else:
                print(f"  ✗ Groq API key not set or invalid")
                return False
    else:
        print(f"  ✗ .env file not found")
        return False
    
    return True

def check_files():
    """Check key files exist"""
    print("\n📄 Checking Key Files...")
    
    base = Path(__file__).parent
    files = {
        'Backend main.py': base / 'backend' / 'main.py',
        'Backend config.py': base / 'backend' / 'config.py',
        'Frontend package.json': base / 'frontend' / 'package.json',
        'Frontend index.html': base / 'frontend' / 'index.html',
        'Frontend App.jsx': base / 'frontend' / 'src' / 'App.jsx',
        'Frontend api.js': base / 'frontend' / 'src' / 'services' / 'api.js',
    }
    
    all_ok = True
    for name, path in files.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name}")
        if not exists:
            all_ok = False
    
    return all_ok

def main():
    print("=" * 60)
    print("🤖 RAG Chatbot System Status Check")
    print("=" * 60)
    
    results = {
        'Directories': check_directories(),
        'Files': check_files(),
        'Configuration': check_config(),
        'Backend': check_backend(),
        'Frontend': check_frontend(),
    }
    
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print("=" * 60)
    
    for component, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {component}")
    
    all_ok = all(results.values())
    
    if all_ok:
        print("\n✅ All systems operational!")
        print("\n🚀 Access your application:")
        print("   Frontend: http://localhost:3000")
        print("   Backend:  http://localhost:8000")
        print("   API Docs: http://localhost:8000/docs")
    else:
        print("\n⚠️ Some issues detected. Please fix them before using the system.")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
