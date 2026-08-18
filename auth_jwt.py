"""JWT decoding for recommender-side authentication.

Laravel signs a short-lived JWT (5 min) with the authenticated user_id
and sends it as Authorization: Bearer <token>. The recommender decodes
the token here and uses the user_id claim — never trusting anything
from the URL path or query string.
"""
from __future__ import annotations

import os

import jwt

JWT_SECRET_ENV = "RECOMMENDER_JWT_SECRET"
JWT_ALGORITHM = "HS256"


def _secret() -> str:
    value = os.environ.get(JWT_SECRET_ENV, "").strip()
    if not value:
        raise RuntimeError(f"{JWT_SECRET_ENV} is required")
    return value


def decode_recommender_jwt(token: str) -> dict:
    """Decode and validate a bearer JWT issued by Laravel.

    Raises ValueError with a short reason on any failure (expired,
    bad signature, missing claim). Callers should translate to
    HTTPException 401.
    """
    try:
        claims = jwt.decode(token, _secret(), algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise ValueError("token expired") from None
    except jwt.InvalidSignatureError:
        raise ValueError("bad signature") from None
    except jwt.InvalidTokenError as exc:
        raise ValueError(f"invalid token: {exc}") from exc

    if "user_id" not in claims:
        raise ValueError("token missing user_id claim")
    try:
        claims["user_id"] = int(claims["user_id"])
    except (TypeError, ValueError):
        raise ValueError("token user_id claim is not an integer") from None
    return claims
