"""Per-IP rate limiting for the recommender FastAPI app.

Default: 60 requests/minute/IP. Callers can override at endpoint level
with @limiter.limit("...") but for this task the app-wide default is enough.
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
