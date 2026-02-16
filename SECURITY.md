# Security Implementation Guide

## Overview

This RAG system implements comprehensive security measures to protect against common web vulnerabilities and ensure safe operation in production environments.

## 🛡️ Security Features

### 1. Rate Limiting

**Implementation:** Token bucket algorithm per IP address

**Limits:**

- Health checks: 120 req/min
- Client management: 60 req/min
- Query/Chat: 30 req/min (expensive operations)
- Document upload: 10 req/min (very expensive)
- Default: 60 req/min

**Headers:**

```
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 29
X-RateLimit-Reset: 1645056789
Retry-After: 45 (when rate limited)
```

**Response (429 Too Many Requests):**

```json
{
  "detail": "Rate limit exceeded. Please try again later.",
  "retry_after": "45"
}
```

### 2. Security Headers

All responses include security headers:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
```

### 3. CORS Configuration

**Development:**

- Allowed origins: `http://localhost:5173`, `http://localhost:3000`
- All credentials supported

**Production:**

- Restricted to specific domains (configure in `main.py`)
- Credentials required
- Limited methods: GET, POST, PUT, DELETE
- Limited headers: Content-Type, Authorization

### 4. Input Validation

**Automatic Detection:**

- SQL injection attempts
- XSS attacks (script tags, event handlers)
- Path traversal (../, %2e%2e)
- Command injection (exec, eval, system)

**Response (400 Bad Request):**

```json
{
  "detail": "Invalid request content"
}
```

### 5. IP Blocking

**Usage:**

```python
# In main.py
blocked_ips = {"192.168.1.100", "10.0.0.50"}
app.add_middleware(SecurityMiddleware, blocked_ips=blocked_ips)
```

**Response (403 Forbidden):**

```json
{
  "detail": "Access forbidden"
}
```

### 6. Environment-Based Configuration

**Development:**

- API docs enabled at `/docs` and `/redoc`
- Permissive CORS for localhost
- Verbose logging

**Production:**

- API docs disabled
- CORS restricted to specific domains
- Trusted host middleware enabled
- Minimal logging

## 🔑 Secrets Management

### Environment Variables

**Never commit `.env` file to Git!**

**Required Variables:**

```bash
# API Keys
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxx

# Database
DATABASE_URL=sqlite:///./rag_system.db

# Admin
ADMIN_PASSWORD=secure_random_password_here

# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### .gitignore

Ensured to exclude:

- `.env` files
- `*.key`, `*.pem`, `*.cert` files
- `secrets.json`
- `vector_stores/` (may contain sensitive data)
- `documents/*.pdf` (user documents)

### Rotating API Keys

1. Generate new key at https://console.groq.com
2. Update `.env` file
3. Restart server: `systemctl restart rag-api` (production)
4. Verify with health check: `curl http://localhost:8000/health`

## 🚨 Security Best Practices

### 1. HTTPS Only (Production)

**Nginx Configuration:**

```nginx
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 2. Database Security

**SQLite (Development):**

- File permissions: `chmod 600 rag_system.db`
- Backup regularly

**PostgreSQL (Production):**

```bash
DATABASE_URL=postgresql://user:password@localhost/rag_db?sslmode=require
```

### 3. File Upload Security

**Current Implementation:**

- Only PDF files accepted
- Max file size: 10MB (configurable in nginx/FastAPI)
- Virus scanning recommended (integrate ClamAV)
- Store in isolated directory

**Recommended Enhancement:**

```python
# In documents router
import magic

def validate_file(file):
    # Check MIME type
    mime = magic.from_buffer(file.read(2048), mime=True)
    if mime != "application/pdf":
        raise HTTPException(400, "Invalid file type")
```

### 4. Logging & Monitoring

**What to Log:**

- Rate limit violations
- Suspicious pattern detections
- Failed authentication attempts
- API errors

**What NOT to Log:**

- API keys
- User passwords
- Full request bodies (may contain PII)

**Setup Log Rotation:**

```bash
# /etc/logrotate.d/rag-api
/var/log/rag-api/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
}
```

### 5. Regular Updates

**Dependencies:**

```bash
# Check for vulnerabilities
pip install safety
safety check

# Update packages
pip list --outdated
pip install --upgrade package_name
```

**Update requirements.txt:**

```bash
pip freeze > requirements.txt
```

## 🔍 Security Audit Checklist

### Pre-Deployment

- [ ] `.env` file not in Git
- [ ] All secrets in environment variables
- [ ] `.gitignore` includes sensitive files
- [ ] API docs disabled in production
- [ ] CORS restricted to specific domains
- [ ] HTTPS certificate installed
- [ ] Rate limiting tested
- [ ] Security headers verified
- [ ] Database credentials secure
- [ ] File upload restrictions in place

### Post-Deployment

- [ ] Monitor rate limit logs
- [ ] Check for suspicious patterns
- [ ] Review access logs daily
- [ ] Update dependencies monthly
- [ ] Rotate API keys quarterly
- [ ] Backup database daily
- [ ] Test disaster recovery

## 🛠️ Incident Response

### Suspected Breach

1. **Immediate Actions:**
   - Rotate all API keys
   - Change admin password
   - Review access logs
   - Block suspicious IPs

2. **Investigation:**

   ```bash
   # Check recent access
   grep "WARNING\|ERROR" /var/log/rag-api/*.log | tail -100

   # Find rate limit violations
   grep "Rate limit exceeded" /var/log/rag-api/*.log

   # Check suspicious patterns
   grep "Suspicious pattern" /var/log/rag-api/*.log
   ```

3. **Recovery:**
   - Restore from backup if data compromised
   - Update security rules
   - Notify affected users
   - Document incident

### Rate Limit Abuse

```python
# Add to security.py blocked_ips set
blocked_ips.add("192.168.1.100")

# Or reload from config file
with open("blocked_ips.txt") as f:
    blocked_ips = set(line.strip() for line in f)
```

## 📊 Security Monitoring

### Metrics to Track

1. **Rate Limit Events:**
   - IPs hitting limits
   - Endpoints being hammered
   - Time patterns (DDoS detection)

2. **Suspicious Activities:**
   - SQL injection attempts
   - XSS attempts
   - Path traversal attempts
   - Unusual query patterns

3. **Performance:**
   - Response times
   - Error rates
   - CPU/Memory usage
   - Database query times

### Alerting

**Setup (example with Prometheus):**

```yaml
# prometheus.yml
scrape_configs:
  - job_name: "rag-api"
    static_configs:
      - targets: ["localhost:8000"]

# Alert rules
groups:
  - name: rag_api_alerts
    rules:
      - alert: HighRateLimitViolations
        expr: rate(rate_limit_violations[5m]) > 10
        for: 2m
        annotations:
          summary: "High rate limit violations detected"
```

## 🔐 Authentication (Future Enhancement)

Currently the system uses environment-based admin password. For production, implement:

### JWT Authentication

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")

# Protect endpoints
@app.post("/api/clients/{client_id}/query")
async def query(client_id: str, request: QueryRequest, user=Depends(verify_token)):
    # Only authenticated users can query
    pass
```

### API Keys per Client

```python
# Generate unique API key per client
import secrets

def generate_api_key():
    return f"rag_{secrets.token_urlsafe(32)}"

# Validate in middleware
api_key = request.headers.get("X-API-Key")
if not validate_api_key(api_key):
    raise HTTPException(401, "Invalid API key")
```

## 📞 Security Contacts

**Report Vulnerabilities:**

- Email: security@yourdomain.com
- PGP Key: [link to public key]

**Security Team:**

- Lead: [Name]
- Response Time: < 24 hours for critical issues

## 📚 References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Rate Limiting Best Practices](https://cloud.google.com/architecture/rate-limiting-strategies-techniques)
- [HTTPS/TLS Configuration](https://ssl-config.mozilla.org/)

---

**Last Updated:** February 16, 2026  
**Version:** 2.0  
**Status:** Production Ready
