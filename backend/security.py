"""
Security Middleware for RAG API

Implements:
- Rate limiting (per IP and per endpoint)
- Security headers
- Request validation
- IP blocking
"""

import time
from collections import defaultdict
from typing import Dict, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timedelta
import re

from logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API endpoints.
    
    Implements per-IP rate limiting with configurable limits per endpoint.
    """
    
    def __init__(self):
        # Store: {ip_address: {endpoint: [(timestamp, tokens_remaining)]}}
        self.requests: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        
        # Rate limit configurations (requests per minute)
        self.limits = {
            "/api/clients/": 60,           # Client management
            "/api/query": 30,               # Query endpoints (expensive)
            "/api/chat": 30,                # Chat endpoints (expensive)
            "/api/documents/upload": 10,    # Document upload (very expensive)
            "/health": 120,                 # Health check (cheap)
            "default": 60                   # Default for other endpoints
        }
        
        # Time window (seconds)
        self.window = 60
        
        # Cleanup interval
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
    def _cleanup_old_requests(self):
        """Remove requests older than the time window."""
        current_time = time.time()
        
        # Only cleanup periodically
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff_time = current_time - self.window
        
        for ip in list(self.requests.keys()):
            for endpoint in list(self.requests[ip].keys()):
                self.requests[ip][endpoint] = [
                    req for req in self.requests[ip][endpoint]
                    if req > cutoff_time
                ]
                
                # Remove empty entries
                if not self.requests[ip][endpoint]:
                    del self.requests[ip][endpoint]
            
            if not self.requests[ip]:
                del self.requests[ip]
        
        self.last_cleanup = current_time
        logger.debug("Rate limiter cleanup completed")
    
    def _get_limit_for_path(self, path: str) -> int:
        """Get rate limit for a specific path."""
        for key, limit in self.limits.items():
            if key in path:
                return limit
        return self.limits["default"]
    
    def check_rate_limit(self, ip: str, path: str) -> tuple[bool, Optional[dict]]:
        """
        Check if request is within rate limits.
        
        Returns:
            (allowed: bool, headers: dict) - allowed status and rate limit headers
        """
        self._cleanup_old_requests()
        
        current_time = time.time()
        limit = self._get_limit_for_path(path)
        
        # Get requests in current window
        requests_in_window = [
            req for req in self.requests[ip][path]
            if req > current_time - self.window
        ]
        
        # Update request history
        self.requests[ip][path] = requests_in_window
        
        # Check limit
        if len(requests_in_window) >= limit:
            remaining = 0
            retry_after = int(self.window - (current_time - min(requests_in_window)))
            allowed = False
        else:
            remaining = limit - len(requests_in_window)
            retry_after = 0
            allowed = True
            # Add current request
            self.requests[ip][path].append(current_time)
        
        # Prepare headers
        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(current_time + self.window)),
        }
        
        if not allowed:
            headers["Retry-After"] = str(retry_after)
        
        return allowed, headers


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive security middleware for FastAPI.
    
    Features:
    - Rate limiting per IP
    - Security headers (HSTS, CSP, etc.)
    - Request validation
    - IP blocking
    """
    
    def __init__(self, app, blocked_ips: Optional[set] = None):
        super().__init__(app)
        self.rate_limiter = RateLimiter()
        self.blocked_ips = blocked_ips or set()
        
        # Suspicious patterns (SQL injection, XSS, path traversal)
        self.suspicious_patterns = [
            re.compile(r"(\bunion\b.*\bselect\b|\bselect\b.*\bfrom\b)", re.IGNORECASE),
            re.compile(r"(<script|javascript:|onerror=|onload=)", re.IGNORECASE),
            re.compile(r"(\.\./|\.\.\\|%2e%2e)", re.IGNORECASE),
            re.compile(r"(exec\(|eval\(|system\(|passthru\()", re.IGNORECASE),
        ]
        
        logger.info("Security middleware initialized with rate limiting")
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request, handling proxies."""
        # Check for proxy headers
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else "unknown"
    
    def _check_suspicious_patterns(self, text: str) -> bool:
        """Check if text contains suspicious patterns."""
        for pattern in self.suspicious_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _add_security_headers(self, response: JSONResponse) -> JSONResponse:
        """Add security headers to response."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        return response
    
    async def dispatch(self, request: Request, call_next):
        """Process each request through security checks."""
        
        # Extract client IP
        client_ip = self._get_client_ip(request)
        path = request.url.path
        
        # Skip security for health check and docs
        if path in ["/health", "/docs", "/openapi.json", "/redoc"]:
            response = await call_next(request)
            return self._add_security_headers(response)
        
        # 1. Check IP blocking
        if client_ip in self.blocked_ips:
            logger.warning(f"Blocked IP attempted access: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Access forbidden"}
            )
        
        # 2. Rate limiting
        allowed, rate_headers = self.rate_limiter.check_rate_limit(client_ip, path)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded: {client_ip} on {path}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded. Please try again later.",
                    "retry_after": rate_headers.get("Retry-After", "60")
                },
                headers=rate_headers
            )
        
        # 3. Check request body for suspicious patterns (for POST/PUT)
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Read body once
                body = await request.body()
                body_str = body.decode("utf-8")
                
                if self._check_suspicious_patterns(body_str):
                    logger.warning(f"Suspicious pattern detected from {client_ip}: {path}")
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": "Invalid request content"}
                    )
                
                # Restore body for downstream processing
                async def receive():
                    return {"type": "http.request", "body": body}
                request._receive = receive
                
            except Exception as e:
                logger.error(f"Error reading request body: {e}")
        
        # 4. Check query parameters
        if request.url.query:
            if self._check_suspicious_patterns(request.url.query):
                logger.warning(f"Suspicious query from {client_ip}: {request.url.query}")
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Invalid query parameters"}
                )
        
        # Log request
        logger.info(f"{request.method} {path} from {client_ip}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Add rate limit headers to response
            for header, value in rate_headers.items():
                response.headers[header] = value
            
            # Add security headers
            response = self._add_security_headers(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )


def get_security_middleware(blocked_ips: Optional[set] = None):
    """Factory function to create security middleware."""
    def middleware_factory(app):
        return SecurityMiddleware(app, blocked_ips)
    return middleware_factory
