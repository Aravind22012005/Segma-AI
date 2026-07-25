"""Thin HTTP client for the FastAPI backend. The frontend never touches
pandas/sklearn directly -- everything goes through this API, exactly like a
judge could with curl."""

from __future__ import annotations

import requests
import streamlit as st


class APIError(Exception):
    pass


def _base_url() -> str:
    return st.session_state.get("backend_url", "http://localhost:8000")


def _get(path: str, **params):
    try:
        r = requests.get(f"{_base_url()}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise APIError(str(e))


def _post(path: str, json_body: dict | None = None, **params):
    try:
        r = requests.post(f"{_base_url()}{path}", params=params, json=json_body, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise APIError(str(e))


def health():
    return _get("/api/health")


def chat(session_id: str, query: str):
    return _post("/api/chat", json_body={"session_id": session_id, "query": query})


def overview():
    return _get("/api/overview")


def segments(session_id: str):
    return _get("/api/segments", session_id=session_id)


def list_customers(session_id: str, tier: str | None = None, limit: int = 200, offset: int = 0):
    kwargs = {"session_id": session_id, "limit": limit, "offset": offset}
    if tier:
        kwargs["tier"] = tier
    return _get("/api/customers", **kwargs)


def customer_detail(customer_id: str, session_id: str):
    return _get(f"/api/customer/{customer_id}", session_id=session_id)


def reset_session(session_id: str):
    return _post("/api/reset", session_id=session_id)


def download_segments_url(session_id: str) -> str:
    return f"{_base_url()}/api/download/segments?session_id={session_id}"
