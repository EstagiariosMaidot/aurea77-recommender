"""Unit tests for auth_jwt.decode_recommender_jwt."""

from __future__ import annotations

import time

import jwt
import pytest

from auth_jwt import JWT_SECRET_ENV, decode_recommender_jwt


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    monkeypatch.setenv(JWT_SECRET_ENV, "supersecret")


def _issue(claims: dict, ttl: int = 60, secret: str = "supersecret") -> str:
    payload = {**claims, "exp": int(time.time()) + ttl, "iat": int(time.time())}
    return jwt.encode(payload, secret, algorithm="HS256")


def test_decodes_valid_token():
    token = _issue({"user_id": 42})
    claims = decode_recommender_jwt(token)
    assert claims["user_id"] == 42


def test_rejects_expired_token():
    token = _issue({"user_id": 42}, ttl=-1)
    with pytest.raises(ValueError, match="expired"):
        decode_recommender_jwt(token)


def test_rejects_bad_signature():
    token = _issue({"user_id": 42}, secret="wrong")
    with pytest.raises(ValueError, match="signature"):
        decode_recommender_jwt(token)


def test_rejects_missing_user_id():
    token = _issue({})
    with pytest.raises(ValueError, match="user_id"):
        decode_recommender_jwt(token)


def test_rejects_non_integer_user_id():
    token = _issue({"user_id": "not-a-number"})
    with pytest.raises(ValueError, match="user_id claim is not an integer"):
        decode_recommender_jwt(token)


def test_coerces_string_integer_user_id():
    """Laravel emits int; but tokens carrying '42' as a string should be accepted."""
    token = _issue({"user_id": "42"})
    claims = decode_recommender_jwt(token)
    assert claims["user_id"] == 42
    assert isinstance(claims["user_id"], int)
