"""FastAPI backend -- the "API" interaction surface + the home of all agent
session state. The Streamlit frontend is a thin client of this API; a judge
could equally well curl these endpoints directly.

Run from the `bank-agent/` directory:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.agent.executor import AgentSession, execute
from backend.agent.explainer import explain
from backend.agent.llm_client import active_provider_name, llm_available
from backend.agent.planner import parse_query
from backend.features import build_unified_customer_view
from backend.json_utils import to_jsonable
from backend.tools import eda_tool

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

app = FastAPI(title="Retail Banking Segmentation & Personalization Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_STATE: dict = {}
SESSIONS: dict[str, AgentSession] = {}


def _load_data():
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    view = build_unified_customer_view(customers, transactions, products)
    _STATE["customers"] = customers
    _STATE["transactions"] = transactions
    _STATE["products"] = products
    _STATE["view"] = view


@app.on_event("startup")
def startup():
    if not (DATA_DIR / "customers.csv").exists():
        raise RuntimeError(f"Dataset not found in {DATA_DIR}. Run `python data/generate_data.py` first.")
    _load_data()


def get_session(session_id: str) -> AgentSession:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = AgentSession(_STATE["view"])
    return SESSIONS[session_id]


class ChatRequest(BaseModel):
    session_id: str
    query: str


class ChatResponse(BaseModel):
    answer: str
    intent: str
    needs_clarification: bool
    plan: dict
    result: dict


@app.get("/api/health")
def health():
    return {"status": "ok", "n_customers": len(_STATE["view"]), "planner": active_provider_name(),
             "llm_active": llm_available()}


@app.post("/api/reset")
def reset(session_id: str):
    SESSIONS[session_id] = AgentSession(_STATE["view"])
    return {"status": "reset"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session = get_session(req.session_id)
    plan = parse_query(req.query)
    exec_result = execute(plan, session)
    answer = explain(req.query, plan, exec_result)
    session.history.append({"query": req.query, "plan": plan, "answer": answer})
    return ChatResponse(
        answer=answer,
        intent=exec_result.get("intent", plan.get("intent", "unknown")),
        needs_clarification=bool(plan.get("needs_clarification")),
        plan=to_jsonable(plan),
        result=to_jsonable(exec_result.get("result", {})),
    )


@app.get("/api/overview")
def overview():
    return eda_tool.dataset_overview(_STATE["customers"], _STATE["transactions"], _STATE["products"])


@app.get("/api/segments")
def segments(session_id: str):
    session = get_session(session_id)
    if not session.has_segmentation():
        return {"segmented": False}
    view = session.view
    payload = {
        "segmented": True,
        "method": session.method,
        "tier_counts": to_jsonable(view["Tier"].value_counts().to_dict()),
    }
    if session.method == "ml" and session.cluster_result is not None:
        coords = session.cluster_result["pca_coords"]
        payload["scatter"] = to_jsonable({
            "customer_id": view["CustomerID"].tolist(),
            "x": coords[:, 0], "y": coords[:, 1],
            "tier": view["Tier"].tolist(), "cluster_id": view["ClusterID"].tolist(),
        })
    return payload


@app.get("/api/customers")
def list_customers(session_id: str, tier: str | None = None, limit: int = 100, offset: int = 0):
    session = get_session(session_id)
    view = session.view
    if tier and "Tier" in view.columns:
        view = view[view["Tier"].str.startswith(tier, na=False)]
    cols = ["CustomerID", "Age", "Gender", "Occupation", "City", "AvgMonthlyBalance",
            "MonthlyTxnFrequency", "CreditScore", "AnnualIncome"]
    if "Tier" in view.columns:
        cols.append("Tier")
    total = len(view)
    page = view[cols].iloc[offset:offset + limit]
    return {"total": total, "customers": to_jsonable(page.to_dict(orient="records"))}


@app.get("/api/customer/{customer_id}")
def customer_detail(customer_id: str, session_id: str):
    session = get_session(session_id)
    row = session.view[session.view["CustomerID"] == customer_id]
    if row.empty:
        raise HTTPException(404, f"Customer {customer_id} not found")
    return to_jsonable(row.drop(columns=["CategorySpendShare"], errors="ignore").iloc[0].to_dict())


@app.get("/api/download/segments")
def download_segments(session_id: str):
    session = get_session(session_id)
    if not session.has_segmentation():
        raise HTTPException(400, "No segmentation has been run yet")
    cols = ["CustomerID", "Tier"]
    if "ClusterID" in session.view.columns:
        cols.append("ClusterID")
    cols += ["AvgMonthlyBalance", "MonthlyTxnFrequency", "AnnualIncome", "CreditScore"]
    csv_buf = io.StringIO()
    session.view[cols].to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    return StreamingResponse(
        iter([csv_buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=customer_segments.csv"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
