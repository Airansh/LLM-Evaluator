# EvalForge — implementation plan

This is the design proposal written before any code, following the build guide.
It is kept up to date as milestones land; deviations are noted inline.

## Directory structure

The repository root *is* the project (no extra `evalforge/` folder):

```text
backend/
  app/
    main.py            FastAPI app factory, CORS, startup schema init
    config.py          pydantic-settings: keys, model IDs, limits (server-side only)
    schemas.py         Pydantic request/response + dataset validation
    api/               routers: health, models, experiments
    db/                SQLAlchemy engine/session, ORM models
    providers/         normalized provider protocol + OpenAI/Anthropic/Gemini adapters
    evaluation/        runner, required-fact scoring, judge, pricing, aggregation, regression
  pricing/             versioned pricing JSON (as-of date per file)
  tests/
frontend/              Next.js + TypeScript + Tailwind + Recharts (Milestone 4)
datasets/customer_support.json
docs/
docker-compose.yml
.env.example
README.md
```

## Data model

All primary keys are UUID strings except where noted. JSON columns hold lists/dicts.

| Entity | Fields | Notes |
| --- | --- | --- |
| **Experiment** | id, name, system_prompt, created_at, baseline_experiment_id → Experiment, regression_threshold (pp, default 3.0), status, dataset_hash, prompt_hash | Prompt and cases are immutable after creation (no update endpoints). `dataset_hash` = SHA-256 of the canonicalized cases, used for baseline compatibility. |
| **TestCase** | pk (int), experiment_id, case_id (user-facing, e.g. `returns_001`), input, context (nullable), expected_answer, required_facts (JSON list), metadata (JSON) | Unique `(experiment_id, case_id)`. A snapshot owned by one experiment. |
| **ModelRun** | id, experiment_id, provider, model_id, status, started_at, completed_at, error, judge_provider, judge_model_id, eval_config (JSON), eval_config_hash, pricing_version | Judge identity and scoring config are stored with each run so comparisons stay interpretable. |
| **EvaluationResult** | id, model_run_id, test_case_pk, response, correctness_score, groundedness_score, required_facts_score, latency_ms, input_tokens, output_tokens, estimated_cost, judge_input_tokens, judge_output_tokens, judge_cost, judge_reason, error, judge_error | Unique `(model_run_id, test_case_pk)`. `NULL` means "not available", never zero. Candidate cost and judge (evaluation) cost are kept separate. |

Statuses: experiment `draft → running → completed | partial | failed`;
model run `pending → running → completed | partial | failed`.

## API contracts

| Method & path | Request | Response |
| --- | --- | --- |
| `GET /health` | – | `{status, version, database}` |
| `GET /models` | – | configured candidate models `{provider, model_id, available, reason}` + judge config. Only booleans for key presence; keys never leave the server. |
| `POST /experiments` | `{name, system_prompt, cases[], baseline_experiment_id?, regression_threshold?}` | experiment detail (201) or 422 with field-level errors |
| `GET /experiments` | – | list with status, case count, latest regression flag |
| `GET /experiments/{id}` | – | detail + per-model aggregates (means with denominators, error counts, costs) |
| `POST /experiments/{id}/run` | `{models: [{provider, model_id}]}` | 202 with created model runs; execution runs in a background task |
| `GET /experiments/{id}/results` | `?case_id=` optional | per-case results incl. judge reason and errors |
| `GET /experiments/{id}/compare` | `?model_id=` optional | per-model paired comparison vs baseline, or `comparable: false` + reasons |

## Implementation sequence

1. **M1 skeleton & contracts** — config, DB + schema init, ORM models, Pydantic schemas,
   `/health`, `/models`, CORS, provider protocol, synthetic dataset, tests.
2. **M2 live run & scoring** — adapters, bounded async runner with retries, judge,
   lexical coverage, pricing, `POST /experiments` + `POST /experiments/{id}/run`, fake providers in tests.
3. **M3 persistence & comparison** — list/detail/results/compare, baseline compatibility, regression threshold.
4. **M4 UI** — dashboard, create, results, failure inspector.
5. **M5 packaging** — Dockerfiles, Compose, README with real results, screenshots.

## Decisions for the choices the guide left open

| Choice | Decision | Why |
| --- | --- | --- |
| Migrations | `Base.metadata.create_all` at startup, no Alembic | Local-first MVP; the schema is small. Alembic is listed as a next step. |
| Sync vs async DB | Sync SQLAlchemy with short sessions; async only for provider I/O | SQLite plus simple writes; avoids aiosqlite complexity. |
| Provider SDKs | Official `openai`, `anthropic`, `google-genai` SDKs inside adapters | Correct usage parsing and error types; provider-specific code stays isolated. |
| Background execution | FastAPI `BackgroundTasks` in-process | Redis and Celery are excluded. Runs interrupted by a restart are marked `failed` on startup. |
| Model IDs | Env vars per provider (comma-separated) | Exact IDs are configured and recorded, not hard-coded. |
| Pricing | `backend/pricing/<as_of>.json`, USD per 1M tokens | Versioned with an as-of date; missing entry → cost "unavailable". |
| Dataset format | JSON object `{name?, description?, cases: [...]}` or a bare array of cases | Matches the guide's sample case. |
| Limits | 1–200 cases, prompt ≤ 8,000 chars, case fields ≤ 20,000 chars | "Sensible" bounds for a small eval set; configurable later. |
| Sample mode (M2) | `SAMPLE_MODE=true` adds `sample/*` models + a heuristic judge, flagged `sample: true` | Lets the pipeline be demoed without keys while never passing mock output off as a live run. |
| Judge output (M2) | Provider-agnostic "JSON only" prompt + Pydantic validation, no native structured-output APIs | One code path for all three providers; malformed output becomes a visible judge error. |
| OpenAI API (M2) | Responses API (`responses.create`) | OpenAI's current primary text API. |
| Initial model IDs (M2) | OpenAI `gpt-6-luna`, `gpt-6-sol` (user-specified); Anthropic `claude-haiku-4-5`, `claude-sonnet-5`; judge `claude-sonnet-5` | OpenAI prices not yet verified → cost shows "unavailable" until added to the pricing file. |
