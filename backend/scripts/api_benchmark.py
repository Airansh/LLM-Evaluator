"""Budget-guarded benchmark that drives a running EvalForge backend over HTTP.

Designed to run as its own instance (a Compose service or a Kubernetes Job) next to the
backend: it needs no database access and no API keys, only the backend's URL. Provider keys
stay in the backend. Standard library only.

Flow: load the backend's sample dataset -> smoke experiment on the first BENCH_SMOKE_CASES
cases -> project the full run's spend per provider from the smoke run's reported tokens ->
run every case only if each projection is under BENCH_BUDGET_USD. Exits non-zero on errors
or when the budget guard refuses.

Environment:
  EVALFORGE_API_URL   backend base URL (default http://backend:8000)
  BENCH_MODELS        comma-separated provider/model_id (default anthropic/claude-haiku-4-5,gemini/gemini-3.8-flash)
  BENCH_MODE          smoke | full | auto (default auto: smoke, then full if within budget)
  BENCH_SMOKE_CASES   cases in the smoke run (default 2)
  BENCH_BUDGET_USD    max projected spend per provider for the full run (default 0.45)
  BENCH_LABEL         text added to experiment names (default "")
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

API = os.environ.get("EVALFORGE_API_URL", "http://backend:8000").rstrip("/")
MODELS = [
    m.strip()
    for m in os.environ.get("BENCH_MODELS", "anthropic/claude-haiku-4-5,gemini/gemini-3.8-flash").split(",")
    if m.strip()
]
MODE = os.environ.get("BENCH_MODE", "auto")
SMOKE_CASES = int(os.environ.get("BENCH_SMOKE_CASES", "2"))
BUDGET = float(os.environ.get("BENCH_BUDGET_USD", "0.45"))
LABEL = os.environ.get("BENCH_LABEL", "").strip()
PROMPT = (
    "You are a customer-support assistant for Northwind Outfitters. Answer using only the policy context "
    "provided. If the context does not answer the question, say so and suggest contacting support."
)
# Used only to bound spend for models without a verified price; never stored as a price.
PESSIMISTIC_PER_MTOK = {"input": 5.0, "output": 30.0}


def call(method: str, path: str, body: Any = None) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{API}{path}", data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {path} -> HTTP {exc.code}: {exc.read().decode()[:500]}") from exc


def wait_for_backend(timeout_s: float = 120) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if call("GET", "/health")["status"] == "ok":
                return
        except (OSError, SystemExit):
            pass
        if time.monotonic() > deadline:
            raise SystemExit(f"backend at {API} did not become healthy")
        time.sleep(2)


def check_models() -> list[dict[str, str]]:
    available = {(m["provider"], m["model_id"]): m for m in call("GET", "/models")["models"]}
    refs, problems = [], []
    for ref in MODELS:
        provider, _, model_id = ref.partition("/")
        info = available.get((provider, model_id))
        if info is None:
            problems.append(f"{ref}: not configured on the backend")
        elif not info["available"]:
            problems.append(f"{ref}: {info['reason']}")
        refs.append({"provider": provider, "model_id": model_id})
    if problems:
        raise SystemExit("models unavailable:\n  " + "\n  ".join(problems))
    return refs


def bound(cost: float | None, tokens_in: int | None, tokens_out: int | None) -> tuple[float, bool]:
    """Measured cost when priced, else a pessimistic bound from reported tokens."""
    if cost is not None:
        return cost, True
    return ((tokens_in or 0) * PESSIMISTIC_PER_MTOK["input"] + (tokens_out or 0) * PESSIMISTIC_PER_MTOK["output"]) / 1e6, False


def spend_by_provider(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    spend: dict[str, dict[str, Any]] = {}
    for case in results["cases"]:
        for r in case["results"]:
            for provider, cost, tin, tout in (
                (r["provider"], r["estimated_cost"], r["input_tokens"], r["output_tokens"]),
                (r["judge_provider"], r["judge_cost"], r["judge_input_tokens"], r["judge_output_tokens"]),
            ):
                if not provider or (tin is None and tout is None):
                    continue
                amount, exact = bound(cost, tin, tout)
                entry = spend.setdefault(provider, {"usd": 0.0, "exact": True})
                entry["usd"] += amount
                entry["exact"] = entry["exact"] and exact
    return spend


def run_experiment(name: str, cases: list[dict[str, Any]], models: list[dict[str, str]]) -> str:
    experiment = call("POST", "/experiments", {"name": name, "system_prompt": PROMPT, "cases": cases})
    call("POST", f"/experiments/{experiment['id']}/run", {"models": models})
    while True:
        detail = call("GET", f"/experiments/{experiment['id']}")
        if detail["status"] != "running":
            return experiment["id"]
        time.sleep(2)


def report(experiment_id: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    detail = call("GET", f"/experiments/{experiment_id}")
    spend = spend_by_provider(call("GET", f"/experiments/{experiment_id}/results"))
    print(f"\n== {detail['name']}  id={detail['id']}  status={detail['status']}  cases={detail['case_count']}")
    for run in detail["runs"]:
        a = run["aggregate"]
        print(
            f"  {run['provider']}/{run['model_id']:<22} [{run['status']}] ok {a['responses_ok']}/{a['total_cases']} "
            f"req_err {a['candidate_errors']} judge_err {a['judge_errors']} | "
            f"correct {a['correctness']['mean']} (n={a['correctness']['n']}) "
            f"grounded {a['groundedness']['mean']} (n={a['groundedness']['n']}) | "
            f"p50 {a['latency_ms_median']} ms | tokens {a['input_tokens']}/{a['output_tokens']} | "
            f"cost {a['candidate_cost']['total']} | judge {run['judge_provider']}/{run['judge_model_id']}"
        )
    for provider, entry in sorted(spend.items()):
        if provider == "sample":
            kind = "fictional sample-mode prices"
        else:
            kind = "measured" if entry["exact"] else "upper bound (no verified price)"
        print(f"  spend {provider:<10} ${entry['usd']:.5f}  ({kind})")
    return detail, spend


def failed(detail: dict[str, Any]) -> bool:
    return any(r["status"] == "failed" or r["aggregate"]["candidate_errors"] for r in detail["runs"])


def main() -> int:
    print(f"EvalForge benchmark runner -> {API}  models={MODELS}  mode={MODE}  budget=${BUDGET}/provider")
    wait_for_backend()
    models = check_models()
    cases = call("GET", "/datasets/sample")["cases"]
    suffix = f" [{LABEL}]" if LABEL else ""

    if MODE in ("smoke", "auto"):
        smoke_id = run_experiment(f"Benchmark smoke ({SMOKE_CASES} cases){suffix}", cases[:SMOKE_CASES], models)
        detail, spend = report(smoke_id)
        if failed(detail):
            print("smoke run had request errors; not continuing")
            return 1
        scale = len(cases) / SMOKE_CASES
        projected = {p: e["usd"] * scale for p, e in spend.items()}
        verdict = all(v < BUDGET for v in projected.values())
        print("  projected full-run spend: " + ", ".join(f"{p} ${v:.4f}" for p, v in sorted(projected.items()))
              + ("  -> within budget" if verdict else f"  -> OVER ${BUDGET}, refusing full run"))
        if MODE == "smoke":
            return 0
        if not verdict:
            return 2

    full_id = run_experiment(f"Benchmark ({len(cases)} cases){suffix}", cases, models)
    detail, _ = report(full_id)
    return 1 if failed(detail) else 0


if __name__ == "__main__":
    sys.exit(main())
