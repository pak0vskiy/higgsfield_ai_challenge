import os
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware


class AuthMiddleware(BaseHTTPMiddleware):
    """Optional bearer token auth. Skipped if MEMORY_AUTH_TOKEN is not set."""

    EXEMPT_PATHS = {"/health"}

    async def dispatch(self, request: Request, call_next):
        # Read token at request time so tests can set/unset the env var dynamically.
        required_token = os.getenv("MEMORY_AUTH_TOKEN", "").strip()
        if not required_token:
            return await call_next(request)
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
        token = auth_header[len("Bearer "):]
        if token != required_token:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")

        return await call_next(request)
