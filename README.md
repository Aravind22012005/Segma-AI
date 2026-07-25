# NeoSegment AI — Customer Segmentation & Personalization Agent

An agent-driven system for retail banking that parses a natural-language
query, runs the right deterministic analytics pipeline (pandas/scikit-learn —
never an LLM doing math), and explains the result in plain language. Built
for the "Customer Segmentation & Personalization Agent for Retail Banking"
hackathon problem statement.

```
Customer.csv   ─┐
Transactions.csv├─► Feature Engine ─► Unified Customer View ─┬─► EDA Tool
Products.csv   ─┘                                            ├─► Segmentation Tool (rule + ML)
                                                               ├─► Explainability Tool
                                                               └─► Recommendation Tool
                                                                        │
User query ─► Planner (LLM or offline rules) ─► Executor (above tools) ─► Explainer ─► Answer
```

## Why it's built this way

- **The dataset is hybrid, not a single Kaggle CSV -- and partly real.**
  `data/generate_data.py` generates 1,500 customers, ~628k transactions
  (24 months), and product ownership from 7 realistic persona archetypes.
  Age, marital status, education, housing/personal loan uptake, and prior
  marketing-campaign response are bootstrapped from 45,211 real rows of the
  **UCI Bank Marketing dataset** (job-matched per archetype); card
  utilization is calibrated to published Kaggle "Credit Card Customers"
  statistics. IBM TabFormer was evaluated but its real transaction file is
  currently undownloadable (IBM's own GitHub LFS quota is exceeded) -- our
  own transaction generator stands in for it. Full provenance --
  exactly which columns are real, calibrated, or synthetic -- is in
  [`data/DATA_SCHEMA.md`](data/DATA_SCHEMA.md).
- **The LLM never touches the numbers.** The planner turns a question into a
  structured intent (`segment_customers`, `aggregate_metric`, `explain_segment_basis`,
  `conversion_candidates`, `recommendation`, EDA intents, …); the executor
  runs that intent through plain pandas/scikit-learn; only the final
  human-readable phrasing optionally goes through an LLM. This keeps results
  deterministic, reproducible, and fast.
- **Works with zero API keys.** If no `GEMINI_API_KEY` / `OPENAI_API_KEY` /
  `ANTHROPIC_API_KEY` is set, the planner falls back to a deterministic
  keyword/regex parser and the explainer falls back to templated text — the
  whole app is fully functional offline. Set a key in `.env` to upgrade the
  planner/explainer to a real LLM (see `.env.example`).
- **Human-in-the-loop.** Ambiguous queries (e.g. "segment customers" with no
  criteria) return a clarifying question instead of guessing.

## Setup

```bash
cd bank-agent
python -m venv venv
venv\Scripts\pip install -r requirements.txt      # Windows
# source venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

python data\generate_data.py                       # generates data/*.csv (~1-2 min)
```

Optional: copy `.env.example` to `.env` and set one LLM API key to enable the
LLM planner/explainer. Not required for the app to work.

## Run

Two terminals (or use `run_backend.bat` / `run_frontend.bat` on Windows):

```bash
# Terminal 1 -- API
venv\Scripts\python -m uvicorn backend.main:app --reload --port 8000

# Terminal 2 -- UI
venv\Scripts\python -m streamlit run frontend/app.py
```

Open the Streamlit URL it prints (default `http://localhost:8501`). The
sidebar shows whether the LLM planner is active or running offline.

You can also talk to the agent directly via the API, no UI required:

```bash
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"session_id":"demo","query":"Segment customers into priority, regular and dormant customers based on balance being maintained and frequency of transactions"}'
```

## Project layout

```
data/
  generate_data.py        data generator (seeded, reproducible; real + calibrated + synthetic)
  uci_calibration.py       bootstraps real fields from the UCI Bank Marketing dataset
  uci_raw/bank-full.csv    real UCI source data (45,211 rows, committed for verifiability)
  DATA_SCHEMA.md           schema + full data provenance (real / calibrated / synthetic per column)
  customers.csv / transactions.csv / products.csv   (agent-facing)
  customers_ground_truth.csv   dev-only, includes hidden persona label

backend/
  features.py              Feature Engine -> Unified Customer View
  tools/
    eda_tool.py             missing values, distributions, correlations, groupby
    feature_tool.py          feature selection for clustering (variance/correlation pruning)
    segmentation_tool.py      rule-based tiering + KMeans clustering (auto-k via silhouette)
    explainability_tool.py    rule thresholds / cluster centroid deviation explanations
    recommendation_tool.py    cross-sell/retention rules + conversion-candidate scoring
  agent/
    llm_client.py            pluggable Gemini/OpenAI/Anthropic client (optional)
    planner.py                NL query -> structured intent (LLM or offline regex/keyword)
    executor.py                runs the intent through the tools above, holds session state
    explainer.py                tool JSON -> human-readable answer (LLM or templated)
  main.py                    FastAPI app (the "API" interaction surface)

frontend/
  app.py                     Streamlit chat + EDA/Segments/Customer/Persona dashboards
  api_client.py                thin HTTP client for the backend
  theme.py                     dark glassmorphic CSS + validated color palette
```

## Example queries to try

- "Segment customers into priority, regular and dormant customers based on balance being maintained and frequency of transactions"
- "On what basis were priority customers selected?"
- "What is the average size of transactions for priority and regular customers?"
- "Which regular customers can be converted to priority? What should be done for them?"
- "Segment customers using ML clustering across income, balance, spend and credit behaviour"
- "Why does customer CUST00001 belong to their segment?"
- "Are there any missing values in the data?"
- "Find customers whose balance has steadily increased"
- "Recommend products for Dormant customers"
