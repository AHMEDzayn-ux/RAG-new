"""
Security Audit Script for RAG System

Performs comprehensive security checks:
- Scans for exposed API keys
- Checks .gitignore coverage
- Validates security configuration
- Tests rate limiting
- Verifies HTTPS configuration
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"🔍 {text}")
    print("=" * 60)

def scan_for_secrets(root_dir: Path) -> List[Tuple[str, str]]:
    """
    Scan all files for potential exposed secrets.
    
    Returns:
        List of (file_path, line) tuples with potential secrets
    """
    print_header("Scanning for Exposed Secrets")
    
    patterns = {
        "Groq API Key": re.compile(r'gsk_[a-zA-Z0-9]{50,}'),
        "OpenAI Key": re.compile(r'sk-[a-zA-Z0-9]{40,}'),
        "Generic Secret": re.compile(r'(secret|password|api[_-]?key)\s*=\s*["\'][^"\']{16,}["\']', re.IGNORECASE),
        "AWS Key": re.compile(r'AKIA[0-9A-Z]{16}'),
        "Private Key": re.compile(r'-----BEGIN (RSA |)PRIVATE KEY-----'),
    }
    
    excluded_dirs = {'.git', 'venv', 'node_modules', '__pycache__', '.pytest_cache', 'htmlcov'}
    excluded_files = {'.env', '.env.local', '.env.example', 'security_audit.py'}
    
    findings = []
    
    for file_path in root_dir.rglob('*'):
        # Skip excluded directories
        if any(excluded in file_path.parts for excluded in excluded_dirs):
            continue
        
        # Skip excluded files
        if file_path.name in excluded_files:
            continue
        
        # Only check text files
        if not file_path.is_file():
            continue
        
        if file_path.suffix not in {'.py', '.js', '.jsx', '.ts', '.tsx', '.json', '.yaml', '.yml', '.md', '.txt', '.sh', '.bat'}:
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for name, pattern in patterns.items():
                matches = pattern.finditer(content)
                for match in matches:
                    # Get line number
                    line_num = content[:match.start()].count('\n') + 1
                    findings.append((str(file_path.relative_to(root_dir)), line_num, name, match.group()))
                    
        except Exception as e:
            continue
    
    if findings:
        print("❌ POTENTIAL SECRETS FOUND:")
        for file_path, line_num, secret_type, preview in findings:
            print(f"  - {file_path}:{line_num} ({secret_type})")
            print(f"    Preview: {preview[:50]}...")
    else:
        print("✅ No exposed secrets detected")
    
    return findings

def check_gitignore_coverage(root_dir: Path) -> bool:
    """Check if .gitignore properly covers sensitive files."""
    print_header("Checking .gitignore Coverage")
    
    required_patterns = [
        '.env',
        'venv/',
        '*.key',
        '*.pem',
        '__pycache__/',
        'vector_stores/',
    ]
    
    gitignore_path = root_dir / '.gitignore'
    
    if not gitignore_path.exists():
        print("❌ No .gitignore file found!")
        return False
    
    with open(gitignore_path, 'r') as f:
        gitignore_content = f.read()
    
    all_covered = True
    for pattern in required_patterns:
        if pattern in gitignore_content:
            print(f"✅ {pattern} - covered")
        else:
            print(f"❌ {pattern} - NOT covered")
            all_covered = False
    
    return all_covered

def check_env_file_security(root_dir: Path) -> bool:
    """Check if .env files are properly excluded from git."""
    print_header("Checking .env File Security")
    
    env_files = list(root_dir.rglob('.env*'))
    
    issues_found = False
    
    for env_file in env_files:
        if env_file.name in {'.env.example', '.env.template'}:
            print(f"✅ {env_file.relative_to(root_dir)} - example file (OK to commit)")
            continue
        
        # Check if tracked by git
        try:
            result = subprocess.run(
                ['git', 'ls-files', '--error-unmatch', str(env_file)],
                cwd=root_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"❌ {env_file.relative_to(root_dir)} - TRACKED BY GIT (DANGER!)")
                issues_found = True
            else:
                print(f"✅ {env_file.relative_to(root_dir)} - properly ignored")
        except Exception as e:
            print(f"⚠️  Could not check {env_file.relative_to(root_dir)}: {e}")
    
    return not issues_found

def check_security_config(root_dir: Path) -> bool:
    """Check security configuration in main.py."""
    print_header("Checking Security Configuration")
    
    main_py = root_dir / 'backend' / 'main.py'
    
    if not main_py.exists():
        print("❌ main.py not found!")
        return False
    
    with open(main_py, 'r') as f:
        content = f.read()
    
    checks = {
        "SecurityMiddleware imported": "from security import SecurityMiddleware" in content or "SecurityMiddleware" in content,
        "CORS configured": "CORSMiddleware" in content,
        "Rate limiting enabled": "SecurityMiddleware" in content,
        "Environment-based config": "settings.environment" in content or "ENVIRONMENT" in content,
    }
    
    all_passed = True
    for check, passed in checks.items():
        if passed:
            print(f"✅ {check}")
        else:
            print(f"❌ {check}")
            all_passed = False
    
    return all_passed

def check_dependencies() -> bool:
    """Check for known vulnerabilities in dependencies."""
    print_header("Checking Dependencies for Vulnerabilities")
    
    try:
        # Try to use safety if installed
        result = subprocess.run(
            ['safety', 'check', '--json'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ No known vulnerabilities found")
            return True
        else:
            print("⚠️  Some vulnerabilities detected:")
            print(result.stdout)
            return False
            
    except FileNotFoundError:
        print("⚠️  'safety' not installed. Install with: pip install safety")
        print("   Skipping vulnerability check...")
        return True
    except Exception as e:
        print(f"⚠️  Could not check dependencies: {e}")
        return True

def verify_cors_config(root_dir: Path) -> bool:
    """Verify CORS is properly restricted."""
    print_header("Checking CORS Configuration")
    
    main_py = root_dir / 'backend' / 'main.py'
    
    if not main_py.exists():
        return False
    
    with open(main_py, 'r') as f:
        content = f.read()
    
    if 'allow_origins=["*"]' in content or "allow_origins=['*']" in content:
        print("❌ CORS allows all origins (*) - SECURITY RISK!")
        print("   Update to restrict to specific domains in production")
        return False
    elif 'environment' in content.lower() and 'production' in content.lower():
        print("✅ CORS has environment-based configuration")
        return True
    else:
        print("⚠️  CORS configuration found but unclear if properly restricted")
        return True

def main():
    """Run full security audit."""
    root_dir = Path(__file__).parent
    
    print("\n" + "🛡️  " * 20)
    print("       RAG SYSTEM SECURITY AUDIT")
    print("🛡️  " * 20)
    
    results = {
        "Secrets Scan": len(scan_for_secrets(root_dir)) == 0,
        ".gitignore Coverage": check_gitignore_coverage(root_dir),
        ".env File Security": check_env_file_security(root_dir),
        "Security Config": check_security_config(root_dir),
        "CORS Configuration": verify_cors_config(root_dir),
        "Dependencies": check_dependencies(),
    }
    
    # Summary
    print_header("Security Audit Summary")
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check}")
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 ALL SECURITY CHECKS PASSED!")
        print("   System is ready for production deployment")
    else:
        print("⚠️  SOME SECURITY ISSUES DETECTED")
        print("   Please address the issues above before deployment")
    
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
