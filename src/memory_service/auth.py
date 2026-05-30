import os
from fastapi import Request
from fastapi.responses import JSONResponse
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
            return JSONResponse(status_code=401, content={"detail": "Missing bearer token"})
        token = auth_header[len("Bearer "):]
        if token != required_token:
            return JSONResponse(status_code=403, content={"detail": "Invalid token"})

        return await call_next(request)
