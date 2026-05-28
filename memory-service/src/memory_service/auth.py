import os
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

_REQUIRED_TOKEN = os.getenv("MEMORY_AUTH_TOKEN", "").strip()

class AuthMiddleware(BaseHTTPMiddleware):
    """Optional bearer token auth. Skipped if MEMORY_AUTH_TOKEN is not set."""

    EXEMPT_PATHS = {"/health"}

    async def dispatch(self, request: Request, call_next):
        if not _REQUIRED_TOKEN:
            return await call_next(request)
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
        token = auth_header[len("Bearer "):]
        if token != _REQUIRED_TOKEN:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")

        return await call_next(request)
