# EvalForge

Run the same evaluation dataset against multiple LLMs (OpenAI, Anthropic, Gemini). Compare quality, latency, token usage and estimated cost, then detect regressions between experiments.

> **Status: Milestone 1 of 5 (backend skeleton and contracts).** No provider calls, scoring or UI yet. No benchmark results have been produced, and none are claimed here. See [`docs/PLAN.md`](docs/PLAN.md) for the design and milestone sequence.

## What works today

- FastAPI app with CORS for the local frontend (`http://localhost:3000` by default).
- `GET /health` checks the app and database.
- `GET /models` lists the configured candidate models and the judge. It shows whether each provider's key is set, but only as a yes/no; keys never leave the server.
- SQLite schema for experiments, immutable test-case snapshots, model runs and evaluation results.
- Pydantic dataset validation: case IDs must be present, well-formed and unique; required fields must be filled; unknown fields are rejected. Limits are 200 cases per dataset and 8,000 characters for the system prompt.
- A normalized async provider protocol (`backend/app/providers/base.py`). The OpenAI, Anthropic and Gemini adapters come in Milestone 2.
- A synthetic customer-support dataset with intentional traps (`datasets/customer_support.json`).

## Quick start (backend)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r backend/requirements-dev.txt
cp .env.example .env          # optional: add keys and exact model IDs
cd backend
uvicorn app.main:app --reload # http://localhost:8000/docs
pytest                        # tests use temporary databases and no API keys
```

## Engineering decisions so far

- **Schema initialization, not migrations.** `create_all` runs at startup and never alters existing tables. After a schema change, delete `backend/data/evalforge.db`. Alembic is a planned next step.
- **"Not available" is `NULL`, never `0`.** Missing token usage, missing prices and failed judge calls are stored as `NULL`. That way they can't pull down an average or look like a real score.
- **Immutable snapshots.** Each experiment stores its own copy of the cases and the prompt, plus a content hash. The hash is how a baseline comparison checks that both sides used the same data.
- **Fact matching is lexical.** Required facts are compared after Unicode NFKC normalization, casefolding and whitespace collapsing. Punctuation is kept, so `$25` does not match `25`.
