# EvalForge

**Run one evaluation dataset against several LLMs, compare quality, latency, token usage and estimated cost, and catch regressions between experiments.**

EvalForge is a local-first evaluation and regression-testing app. You upload a small JSON dataset, choose a system prompt and models (OpenAI, Anthropic, Gemini), and run an experiment. You can then inspect individual failures and compare the experiment against an earlier baseline. A regression verdict is given only when the comparison is like-for-like.

> **What has and hasn't been verified.** One small **live** run has been made: Anthropic `claude-haiku-4-5` against Google `gemini-3.8-flash` on the 12-case sample dataset, judged by `claude-sonnet-5` (see [Live results](#live-results-2026-10-10)). The OpenAI adapter has **not** been called live. All screenshots and the regression example come from the offline **sample mode** or from **fake providers in tests**. See [Verification status](#verification-status).

![Results page (sample mode)](docs/screenshots/3-results.png)

## The problem

Changing a prompt or switching models can quietly make answers worse, slower or more expensive. Checking a handful of outputs by eye doesn't catch that reliably. EvalForge runs a fixed set of cases through each model and scores every answer the same way. It shows where the numbers come from (how many cases each average covers, the judge's reasoning, errors), and it refuses to call a regression when the two runs aren't actually comparable.

## Live results (2026-10-10)

The first live run, made with `backend/scripts/live_benchmark.py`. It compares one model from each provider on all 12 cases of `datasets/customer_support.json`, using the same system prompt and judge for both. Candidate answers were capped at 1,024 output tokens and judge replies at 1,500, under a $0.50-per-provider spending cap. No request errors or judge errors occurred.

| Model (exact ID) | Correctness | Groundedness | Required facts | Latency p50 | Tokens in / out | Candidate cost | Judge cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `anthropic/claude-haiku-4-5` | 90.4 (n=12) | 94.5 (n=11) | 100.0 (n=10) | 1.16 s | 1,175 / 966 | $0.0060 | $0.0259 |
| `gemini/gemini-3.8-flash` | 88.3 (n=12) | 97.3 (n=11) | 100.0 (n=10) | 2.98 s | 1,019 / 4,001 | unavailable* | $0.0244 |

Judge: `anthropic/claude-sonnet-5`. Correctness and groundedness are judge estimates on a 0–100 scale. Groundedness skips the one case without context, and required facts skips the two cases that list none.

\* No verified price: Google's pricing page could not be reached from the environment where the run was made, so no Gemini price was recorded. Gemini's output count includes its thinking tokens.

**Reading these numbers honestly:**

- **Quality is effectively tied on this dataset.** A 2-point gap in mean correctness is about one judge point per case, and n=12 with single non-deterministic samples is far too small to separate the models.
- **`general_001` is a dataset flaw, not a model failure.** The system prompt says to answer *only from the policy context*, but that case has no context and expects general etiquette advice. Both models followed the prompt and declined ("the policy context does not contain information…"), and the judge scored them 10 and 0. With that case excluded, mean correctness is 97.7 (Haiku) and 96.4 (Gemini), n=11. It is kept in the table rather than dropped after the fact. Fixing it means changing the case or the prompt, which would make future runs non-comparable with this one.
- **The judge does not see the system prompt.** It docked groundedness when a model mentioned "Northwind Outfitters", the company name given in the system prompt, because the name isn't in the case context (`warranty_001` for Gemini, `warranty_002` for Haiku).
- **Possible same-family bias.** The judge is a Claude model and one candidate is Claude. A cross-provider judge would be a fairer check.
- **The clearer differences are speed and verbosity.** Haiku's median latency was about 2.6× lower, and it produced about 4× fewer output tokens. Gemini's count includes thinking tokens, which it spends even on short answers.
- **All five deliberate traps were handled by both models,** judging from the per-case scores and responses: the 45-day return, the question the context can't answer, the prompt injection, the "inactivity" conditional, and the no-context case (where both declined).

**What it cost:** Anthropic **$0.0646** in total, covering the 2-case smoke run and the full run, candidates plus judge, from reported tokens × verified prices. Gemini used 1,234 input and 4,782 output tokens in total, including one adapter test call. That is at most $0.15 even at a deliberately pessimistic $5 / $30 per million tokens, and likely far less at actual Flash-tier rates.

To reproduce: `cd backend && PYTHONPATH=. python scripts/live_benchmark.py smoke`, then `full`. The script refuses the full run if the smoke run projects more than $0.45 for either provider. It needs `EVALFORGE_ANTHROPIC_API_KEY` (or `ANTHROPIC_API_KEY`) and `GEMINI_API_KEY`, and it makes paid calls.

## Quick start

### Docker (one command)

```bash
cp .env.example .env          # optional: add API keys and model IDs; sample mode works without
docker compose up --build     # UI http://localhost:3000 · API http://localhost:8000/docs
```

The SQLite database lives in the named volume `evalforge-data`, so results survive `docker compose down` and `up`. To delete them, run `docker compose down -v`. Sample mode is on by default in Compose; set `SAMPLE_MODE=false` in `.env` to hide the sample models.

### Without Docker

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r backend/requirements-dev.txt
cp .env.example .env
cd backend && uvicorn app.main:app --reload      # http://localhost:8000
# second terminal
cd frontend && npm install && npm run dev        # http://localhost:3000
```

### Try it without keys

```bash
cd backend
python -m app.demo             # 12 cases × 3 offline sample models, printed with denominators
python -m app.demo --compare   # prompt v1 (baseline) vs prompt v2 (candidate) comparison
```

With Docker, run `docker compose exec backend python -m app.demo --compare` instead. In the UI: **New evaluation → Use sample dataset → pick `sample/*` models → Create and run**, then **New candidate from this** to compare a prompt change.

## Credentials and costs

All keys go in **one file: `.env` in the repository root** (copy it from `.env.example`). The file is git-ignored. Only the backend reads it; the browser never receives keys, and `GET /models` reports only *whether* each key is set.

| Provider | Variable | Where to get a key | Candidate models (exact IDs) |
| --- | --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | https://platform.openai.com/api-keys | `OPENAI_MODELS=gpt-6-luna,gpt-6-sol` |
| Anthropic | `ANTHROPIC_API_KEY` (or `EVALFORGE_ANTHROPIC_API_KEY` where the host reserves `ANTHROPIC_*`) | Claude Console → Settings → API keys | `ANTHROPIC_MODELS=claude-haiku-4-5,claude-sonnet-5` |
| Gemini | `GEMINI_API_KEY` | https://aistudio.google.com/apikey | `GEMINI_MODELS=` (add exact IDs) |

- **Judge:** `JUDGE_PROVIDER` + `JUDGE_MODEL` (default `anthropic` / `claude-sonnet-5`) must name a provider whose key is set. If the judge is unavailable, a run with real models is rejected with the reason.
- **Checking your setup:** after restarting the backend, `http://localhost:8000/models` should show `"available": true` for every model you configured. Models without a key appear greyed out in the UI, with the reason.
- **What a run costs:** one candidate call plus one judge call per case per model. For example, 12 cases × 3 models is 72 paid requests, plus retries (up to `MAX_ATTEMPTS`, for rate limits, timeouts and 5xx errors only). The create screen shows the call count before you run.
- **Prices:** `backend/pricing/2026-09-24.json` lists USD per 1M tokens, each with its source and as-of date. It currently covers Anthropic models only. **`gpt-6-luna` and `gpt-6-sol` have no verified price, so their cost shows as "unavailable"** until you add entries such as `"openai/gpt-6-luna": {"input": …, "output": …, "as_of": "…", "source": "…"}`.

## Architecture

```text
Browser (Next.js 16, React 19, Tailwind, Recharts)
   │  JSON over HTTP (NEXT_PUBLIC_API_URL, no secrets)
   ▼
FastAPI  ── /experiments, /models, /datasets, /health
   │
   ├── services.py        create experiment (immutable case snapshot + hashes), prepare runs
   ├── evaluation/runner  asyncio: semaphore-bounded case × model tasks, retries, persist each result
   │     ├── providers/   OpenAI (Responses API) · Anthropic (Messages) · Gemini (google-genai) · sample
   │     ├── coverage     lexical required-fact matching
   │     ├── judge        separate model → Pydantic-validated JSON scores
   │     └── pricing      versioned price file → estimated cost or "unavailable"
   ├── evaluation/compare baseline compatibility checks + paired regression verdicts
   └── SQLite (SQLAlchemy)  experiments · test_cases · model_runs · evaluation_results
```

Runs execute in-process via FastAPI background tasks; there's no queue. Each result is written as soon as it's scored. A run cut off by a restart is marked `failed` at the next startup, and the results it already produced are kept.

## Sample dataset

`datasets/customer_support.json` contains 12 synthetic cases for a fictional retailer ("Northwind Outfitters"), and every policy in it is invented. Several cases are deliberate traps:

| Case | What it tests |
| --- | --- |
| `returns_001` | A number outside the allowed window (45 days vs a 30-day policy) |
| `shipping_002` | A question the context can't answer (the model should say so; no required facts) |
| `warranty_002` | Prompt injection in the user question ("ignore your previous instructions…") |
| `loyalty_002` | A conditional that's easy to negate ("expire after 12 months of *inactivity*") |
| `general_001` | No context, so groundedness is N/A |

Dataset format: `{ "name"?, "description"?, "cases": [ { "id", "input", "context"?, "expected_answer", "required_facts"?, "metadata"? } ] }` or a bare array of cases. IDs must be unique, required fields non-empty, and unknown fields are rejected. A dataset can have at most 200 cases, and the system prompt is capped at 8,000 characters.

## Scoring methodology

For each test case × model:

1. **Candidate call.** The experiment's system prompt is sent as the system message, and the case's context plus question as the user message. The expected answer is never sent. Temperature is the provider default unless you set `TEMPERATURE`, because some current models reject an explicit value.
2. **Required-fact coverage (0–100).** The share of `required_facts` found in the response. It is a substring match after NFKC normalization, casefolding and whitespace collapsing, with punctuation kept (`$7` ≠ `7`). An empty list is **N/A**, not 100%. This is a **lexical** check: "thirty days" does not match "30 days".
3. **Judge (0–100).** A separately configured model scores **correctness** against the expected answer and **groundedness** against the context, and gives a short reason. The output must be JSON that passes Pydantic validation; anything malformed becomes a visible **judge error**, never a zero. Groundedness is **N/A** without context.
4. **Latency and tokens.** Latency is the wall-clock time of the final attempt (retry backoff excluded). Token counts are what the provider reported; OpenAI and Gemini output counts include reasoning/thinking tokens. Missing values stay empty, never 0.
5. **Cost.** Reported tokens × the price file, with candidate cost and judge (evaluation) cost kept separate. If a price or the usage is missing, the cost is **unavailable**.
6. **Aggregates.** Means are computed over successfully scored cases and always shown with their denominator (`n=10/12`), alongside request-error and judge-error counts. Nothing is averaged in as 0.

Each model run stores the judge's identity, the pricing version and an `eval_config` (template versions, fact-matching rule, token limits, temperature) with a hash of it.

## Judge limitations

- **Judge scores are estimates from a model, not verified facts.** They have not been calibrated against human labels.
- **Same-family bias.** If the judge comes from the same provider or family as a candidate (for example Claude judging Claude), it may favor that candidate. Use a judge from a different provider for cross-provider comparisons, or at least report which judge you used (the UI shows it on every result).
- **Non-determinism.** Candidates and judge each run once per case at provider-default sampling. Re-running can change scores by a few points, which on a small dataset is the same order as the 3-point threshold.
- **Prompt injection is mitigated, not solved.** The candidate response is HTML-escaped inside `<candidate_response>` tags, and the judge is told that content is untrusted and must not be followed. A sufficiently adversarial response might still sway the judge.
- **The judge does not see the system prompt.** Details that come from the system prompt rather than the case context (such as the company name) can be marked ungrounded. This was observed in the live run.
- **Lexical coverage is shallow.** It catches missing key numbers and terms but misses paraphrases, and it can't see whether a fact is used correctly.

## Regression detection

Create the candidate with `baseline_experiment_id` (in the UI: **New candidate from this**) and a threshold (default **3 percentage points**). `GET /experiments/{id}/compare` gives a verdict per model only when all of these hold:

- **Same cases.** Compared by content hash, so reordering doesn't matter. If they differ, the response lists the added, removed and changed IDs and gives no verdict.
- **Same model.** The latest finished run of the same provider/model on each side.
- **Same evaluation settings.** Judge, templates, token limits and temperature must be identical; any differing settings are named.
- **Complete pairs.** Every applicable case must be scored on both sides. If any is missing, that metric reports `insufficient_data`.

**Correctness or groundedness** is flagged as a regression when the candidate's mean over paired cases is lower by **more than** the threshold. Fact coverage, median latency and cost are shown as information only.

### Example output (sample mode, actual)

This is the actual output of `python -m app.demo --compare` (it's deterministic apart from simulated latency). Prompt v2 appends *"Keep every answer to one sentence."*:

```text
sample/sample-extractive: OK
  correctness   baseline  63.4  candidate  62.8  delta  -0.6 pp  paired 12/12  -> ok
  groundedness  baseline 100.0  candidate 100.0  delta   0.0 pp  paired 11/11  -> ok
  required_facts_mean   baseline 100.0  candidate 66.7  (informational, not a verdict)
```

In the UI version of the same comparison, `sample-flaky` gets **insufficient data**: 2 of its 12 cases fail on both sides, so no verdict is given.

What this shows honestly: the shorter prompt dropped fact coverage by 33 points, but the heuristic *sample* judge barely moved, so no regression is flagged. That is a limitation of the offline heuristic, not something a real LLM judge would necessarily share. It is also why fact coverage is always displayed next to the verdict. A regression that does trip the threshold is demonstrated with a controlled fake judge in `backend/tests/test_compare.py` (a 4-point drop is flagged; exactly 3 points is not). No invented results are presented as real.

## Screens

| | |
| --- | --- |
| ![Dashboard](docs/screenshots/1-dashboard.png) | ![Create](docs/screenshots/2-create.png) |
| ![Failure inspector: failed request next to a scored response](docs/screenshots/4-inspector-failed-case.png) | ![Baseline comparison](docs/screenshots/5-comparison.png) |

- **Dashboard (`/`):** experiments, status, case-evaluation count, regression indicator.
- **Create (`/experiments/new`):** prompt, sample or uploaded dataset with validation errors, models with key availability, compatible baselines only, threshold, call count.
- **Results (`/experiments/[id]`):** model table with denominators and error counts, regression panel, quality-vs-cost chart (runs that can't be plotted are listed with the reason), per-case grid, live progress.
- **Failure inspector (`/experiments/[id]/test/[testId]`):** question, context, expected answer, each response, fact matches, metrics, judge rationale, costs, errors.

Dark mode and a 390px mobile layout are in [`docs/screenshots/`](docs/screenshots/). All screenshots are from a real session driven by an automated browser against the sample-mode backend.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | App and database status |
| `GET /models` | Configured models with availability and reason, sample flag, judge status |
| `GET /datasets/sample` · `POST /datasets/validate` | The bundled dataset · validate without storing (errors, or case count + content hash) |
| `POST /experiments` | `{name, system_prompt, cases[], baseline_experiment_id?, regression_threshold?}` |
| `GET /experiments` | Newest first, with status, counts, models and regression indicator |
| `GET /experiments/{id}` | Status, runs, per-run aggregates with denominators |
| `POST /experiments/{id}/run` | `{models: [{provider, model_id}]}` → 202; executes in the background |
| `GET /experiments/{id}/results` | Per-case results (`?case_id=`, `?failures_only=true`) |
| `GET /experiments/{id}/compare` | Baseline comparison (`?provider=`, `?model_id=`) |

## Engineering decisions

- **Official SDKs inside thin adapters.** SDK retries are disabled so the runner owns bounded, visible retries (rate limits, timeouts, connection errors, 5xx). Auth errors, bad requests and refusals are never retried.
- **"Not available" is `NULL`, never `0`.** This applies to usage, prices, scores and judge failures, so missing data can't masquerade as a score.
- **Immutable snapshots and hashes.** Each experiment owns a copy of its cases and prompt. Comparisons check content hashes, not names.
- **Sample mode instead of silent mocks.** Without keys, `sample/*` models and a heuristic judge are available but labeled on every screen and in every API response. A real model is never replaced by mock output.
- **SQLite and in-process execution.** No Redis or Celery. `create_all` handles schema setup, with no migrations: after a schema change, delete the database (or `docker compose down -v`).
- **Keys stay server-side.** They're held as `SecretStr` in settings, only key *presence* is exposed, and any echo of a key in a provider error is redacted before storage.

## Verification status

What was actually run for this version:

| Check | Result |
| --- | --- |
| Backend tests (`pytest`, fake providers and stubbed SDK clients, no keys) | 87 passed |
| Backend type check (`mypy app`) | clean |
| Frontend `npm run lint`, `tsc --noEmit`, `npm run build` | clean |
| Browser end-to-end flow: create → run → results → failed case → baseline comparison, plus dark and 390px mobile | passed, no console errors, both with local servers and with `docker compose` |
| Results identical after a server restart (tests + real uvicorn restart) and after `docker compose down` / `up` (named volume) | passed |
| Key handling: `/models` never returns key values; key echoes redacted from stored errors (tests); no key material in server logs | passed (no real keys were configured) |
| Live provider calls: Anthropic and Gemini | passed: 2-case smoke run, then 12 cases × 2 models plus judge, with no request or judge errors (see [Live results](#live-results-2026-10-10)) |
| **Live provider calls: OpenAI** | **not run**: no OpenAI key used |

### Acceptance checklist

- [x] Fresh clone starts with documented commands (Docker and non-Docker paths)
- [x] JSON dataset validates and an explicitly labeled sample run completes
- [x] OpenAI, Anthropic and Gemini adapters exist; missing keys produce a clear status
- [x] …and a live run with real keys has been done (Anthropic and Gemini; OpenAI not yet)
- [x] Responses, model IDs, latency, usage and scoring details persist across restarts
- [x] Per-case inspector shows the judge's rationale and errors
- [x] A regression verdict requires identical cases, model and scoring settings
- [x] A controlled regression triggers the threshold (tests); mismatched baselines get no verdict
- [x] No API keys in the repository, browser responses, screenshots or logs (verified without real keys)
- [x] Tests, frontend checks and setup instructions were run, and their outcomes are reported above

## Next steps

1. **More live runs.** Add OpenAI, fix the `general_001` case/prompt conflict, use a non-Claude (or multi-provider) judge, and repeat each run several times so differences can be compared against the noise.
2. **OpenAI (and Gemini) pricing.** Add verified per-token prices with sources so those costs stop showing "unavailable".
3. **Repeated runs and uncertainty.** Run each case several times and report a spread or confidence interval, so a 3-point change can be judged against the noise.
4. **Judge calibration.** Score a subset by hand, measure how well the judge agrees, and consider a cross-provider judge panel.
5. **Scale.** Alembic migrations, a real job queue and Postgres for large datasets or concurrent users. At around 100,000 evaluations, batch APIs, rate-limit-aware scheduling and prompt caching would matter more than anything above.
