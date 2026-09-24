# EvalForge

Run the same evaluation dataset against multiple LLMs (OpenAI, Anthropic, Gemini). Compare quality, latency, token usage and estimated cost, then detect regressions between experiments.

> **Status: Milestone 2 of 5 (live-run pipeline and scoring).** The backend runs experiments end to end. So far that has only been verified with **fake providers in tests** and the **offline sample mode**. **No live provider call has been made yet**, so this README contains no real model results. Comparison endpoints and the UI come next. See [`docs/PLAN.md`](docs/PLAN.md).

## Quick start (backend, no keys needed)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r backend/requirements-dev.txt
cp .env.example .env           # sample mode is on by default in the example
cd backend
python -m app.demo             # SAMPLE setup run: 12 cases × 3 offline sample models
uvicorn app.main:app --reload  # API at http://localhost:8000, interactive docs at /docs
pytest                         # temporary databases, fake providers, no API keys
```

`python -m app.demo` runs the real runner, scoring and SQLite persistence against the offline **sample** models, then prints per-model aggregates with denominators. Sample output is text heuristics, simulated latency and fictional prices, and it is labeled `SAMPLE` everywhere. It shows the pipeline works; it tells you nothing about any real model.

## Adding your API keys

Keys live in **one file: `.env` in the repository root** (next to `.env.example`). The file is git-ignored. Only the backend reads it, and `GET /models` reports only *whether* each key is set, never the value.

1. Create the file if you haven't: `cp .env.example .env`
2. Paste in the keys you have. Any provider you leave blank shows up as unavailable, with the reason.

   | Provider | Variable | Where to get a key | Candidate models |
   | --- | --- | --- | --- |
   | OpenAI | `OPENAI_API_KEY` | https://platform.openai.com/api-keys | `OPENAI_MODELS=gpt-6-luna,gpt-6-sol` |
   | Anthropic | `ANTHROPIC_API_KEY` | Claude Console → Settings → API keys | `ANTHROPIC_MODELS=claude-haiku-4-5,claude-sonnet-5` |
   | Gemini | `GEMINI_API_KEY` | https://aistudio.google.com/apikey | `GEMINI_MODELS=` (add exact IDs) |

3. Choose the judge. `JUDGE_PROVIDER` + `JUDGE_MODEL` (default `anthropic` / `claude-sonnet-5`) must name a provider whose key is set. A real model can't run without an available judge; the API rejects the request and says why.
4. Restart the backend and check `http://localhost:8000/models`. Every model you configured should show `"available": true`.

Alternatively, export the same variables in your shell or container environment. Real environment variables take precedence over `.env`.

**Costs:** every live run makes one candidate call plus one judge call per case per model, so 12 cases × 3 models is 72 paid requests. Retries (up to `MAX_ATTEMPTS`, for rate limits, timeouts and 5xx errors only) can add more.

## API (so far)

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | App and database status |
| `GET /models` | Configured models, availability and reason, sample flag, judge status |
| `POST /experiments` | Create an experiment from `{name, system_prompt, cases[], baseline_experiment_id?, regression_threshold?}` |
| `GET /experiments/{id}` | Status, model runs, per-run aggregates with denominators |
| `POST /experiments/{id}/run` | `{models: [{provider, model_id}]}`. Validates, returns 202, and runs in the background |

## How a run is scored

For each test case × selected model:

1. **Candidate call.** The system prompt is sent as the system message, and the context plus question as the user message. The expected answer is never sent. Concurrency is capped at `MAX_CONCURRENCY`. Rate limits, timeouts, connection errors and 5xx get up to `MAX_ATTEMPTS` tries with exponential backoff and jitter. Auth errors, bad requests and refusals are not retried.
2. **Required-fact coverage** is the percent of `required_facts` found in the response. It is a substring match after NFKC normalization, casefolding and whitespace collapsing, and punctuation is kept. An empty list is **N/A**, not 100%. This is a **lexical** check, so "thirty days" does not match "30 days".
3. **Judge.** A separately configured model returns JSON `{correctness, groundedness, reason}` on a 0–100 scale. The output is validated with Pydantic. Malformed output becomes a visible **judge error**, never a zero score. Groundedness is **N/A** when the case has no context. The candidate response is HTML-escaped inside `<candidate_response>` tags, and the judge is told that content is untrusted data. Judge scores are estimates, not facts.
4. **Latency** is the wall-clock time of the final attempt; retry backoff is excluded. **Tokens** are what the provider reported (OpenAI and Gemini output counts include reasoning/thinking tokens). Missing usage stays empty, never 0.
5. **Cost** is tokens × the price in `backend/pricing/<version>.json`. Each price records its source and as-of date. **If a model has no price entry, its cost shows as unavailable.** The file currently lists Anthropic prices only. **`gpt-6-luna` and `gpt-6-sol` have no verified price yet**; add them as `"openai/gpt-6-luna": {"input": …, "output": …, "as_of": "…", "source": "…"}` in USD per 1M tokens. Candidate cost and judge cost are kept separate.
6. **Persistence.** Each result is saved as soon as it is scored. A failing model or case is recorded with its error and does not erase other results. A run is `completed`, `partial` or `failed`. Runs cut short by a server restart are marked `failed` at the next startup, and the results they already produced are kept.

Each model run stores its judge identity, pricing version and an `eval_config` (prompt template versions, fact-matching rule, token limits, temperature) along with a hash of it. Milestone 3 uses that hash to refuse comparisons across different scoring setups.

## Engineering decisions so far

- **Official SDKs inside thin adapters** (`backend/app/providers/`). The SDKs' built-in retries are disabled so the runner owns bounded, visible retries. Provider errors are normalized into a single set of error kinds.
- **Temperature is opt-in.** Current models such as Claude Sonnet 5 reject an explicit `temperature`. Outputs are therefore non-deterministic by default, which is one more reason not to over-read small differences.
- **Schema initialization, not migrations.** `create_all` runs at startup and never alters existing tables. After a schema change, delete `backend/data/evalforge.db`. Alembic is a planned next step.
- **"Not available" is `NULL`, never `0`.** Missing usage, missing prices and failed judge calls can't pull averages down or look like real scores. Aggregates report `n` for every mean.
- **Immutable snapshots.** Each experiment stores its own copy of the cases and the prompt, plus content hashes.
- **Keys never leave the backend.** API keys are masked in settings (`SecretStr`), and any echo of a key in a provider error is redacted before it is stored.
