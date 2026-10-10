"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";

import { QualityCostChart } from "@/components/QualityCostChart";
import { Badge, ButtonLink, Card, Empty, ErrorState, Loading, PageHeader, SampleBadge, StatusBadge, VerdictBadge } from "@/components/ui";
import { api, type CaseOut, type ComparisonOut, type ExperimentOut, type RunOut } from "@/lib/api";
import { dateTime, modelLabel, ms, pp, score, tokens, usd } from "@/lib/format";
import { useApi } from "@/lib/useApi";

const POLL_MS = 2000;

function isActive(experiment: ExperimentOut): boolean {
  return experiment.status === "running" || experiment.runs.some((r) => r.status === "pending" || r.status === "running");
}

function ScoreCell({ mean, n, total }: { mean: number | null; n: number; total: number }) {
  return (
    <div className="min-w-24">
      <div className="flex items-baseline justify-between gap-2 tabular-nums">
        <span className="font-medium">{score(mean)}</span>
        <span className="text-xs text-ink-3">n={n}/{total}</span>
      </div>
      {mean != null && (
        <div className="mt-1 h-1.5 w-full rounded-full bg-sunken" aria-hidden="true">
          <div className="h-1.5 rounded-full bg-accent" style={{ width: `${Math.max(0, Math.min(100, mean))}%` }} />
        </div>
      )}
    </div>
  );
}

function ModelTable({ experiment }: { experiment: ExperimentOut }) {
  if (experiment.runs.length === 0) {
    return <p className="text-sm text-ink-2">No model runs yet.</p>;
  }
  return (
    <div className="-m-4 overflow-x-auto">
      <table className="w-full min-w-[900px] text-left text-sm">
        <caption className="sr-only">Per-model aggregates; means are over successfully scored cases, shown with their denominators.</caption>
        <thead className="bg-sunken text-xs uppercase tracking-wide text-ink-3">
          <tr>
            <th scope="col" className="px-4 py-2 font-medium">Model</th>
            <th scope="col" className="px-4 py-2 font-medium">Status</th>
            <th scope="col" className="px-4 py-2 font-medium">Responses</th>
            <th scope="col" className="px-4 py-2 font-medium">Correctness</th>
            <th scope="col" className="px-4 py-2 font-medium">Groundedness</th>
            <th scope="col" className="px-4 py-2 font-medium">Required facts</th>
            <th scope="col" className="px-4 py-2 font-medium">Latency p50</th>
            <th scope="col" className="px-4 py-2 font-medium">Tokens in / out</th>
            <th scope="col" className="px-4 py-2 font-medium">Est. cost</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line align-top">
          {experiment.runs.map((run) => {
            const agg = run.aggregate;
            return (
              <tr key={run.id}>
                <th scope="row" className="px-4 py-3 text-left font-medium">
                  <div className="flex flex-wrap items-center gap-2">
                    {modelLabel(run.provider, run.model_id)}
                    {run.sample && <SampleBadge />}
                  </div>
                  <div className="mt-0.5 text-xs font-normal text-ink-3">
                    judge {run.judge_provider}/{run.judge_model_id}
                  </div>
                  {run.error && <div className="mt-1 max-w-64 text-xs font-normal text-critical">{run.error}</div>}
                </th>
                <td className="px-4 py-3"><StatusBadge status={run.status} /></td>
                <td className="px-4 py-3 tabular-nums">
                  {agg ? (
                    <>
                      <div>{agg.responses_ok}/{experiment.case_count} ok</div>
                      {agg.candidate_errors > 0 && <div className="text-xs text-critical">✕ {agg.candidate_errors} request errors</div>}
                      {agg.judge_errors > 0 && <div className="text-xs text-warning">! {agg.judge_errors} judge errors</div>}
                    </>
                  ) : "–"}
                </td>
                <td className="px-4 py-3">{agg && <ScoreCell {...agg.correctness} total={experiment.case_count} />}</td>
                <td className="px-4 py-3">{agg && <ScoreCell {...agg.groundedness} total={experiment.case_count} />}</td>
                <td className="px-4 py-3">{agg && <ScoreCell {...agg.required_facts} total={experiment.case_count} />}</td>
                <td className="px-4 py-3 tabular-nums">{ms(agg?.latency_ms_median)}</td>
                <td className="px-4 py-3 tabular-nums text-ink-2">{tokens(agg?.input_tokens)} / {tokens(agg?.output_tokens)}</td>
                <td className="px-4 py-3 tabular-nums">
                  {usd(agg?.candidate_cost.total)}
                  {agg && agg.candidate_cost.total != null && !agg.candidate_cost.complete && (
                    <div className="text-xs text-warning">! partial ({agg.candidate_cost.n_priced} priced)</div>
                  )}
                  <div className="text-xs text-ink-3">+ judge {usd(agg?.judge_cost.total)}</div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function RegressionPanel({ experiment }: { experiment: ExperimentOut }) {
  const active = isActive(experiment);
  const { data, error, loading, reload } = useApi<ComparisonOut>(
    () => api.compare(experiment.id),
    [experiment.id, active, experiment.runs.length],
  );

  if (!experiment.baseline_experiment_id) {
    return (
      <div className="text-sm text-ink-2">
        <p>No baseline is set, so there is nothing to compare against.</p>
        <p className="mt-2">
          To test a prompt change, start a candidate from this experiment: it reuses these exact cases and models, and you edit the prompt.
        </p>
        <div className="mt-3">
          <ButtonLink href={`/experiments/new?baseline=${experiment.id}`} variant="secondary">New candidate from this experiment</ButtonLink>
        </div>
      </div>
    );
  }
  if (loading && !data) return <Loading label="Comparing…" />;
  if (error) return <ErrorState error={error} onRetry={reload} />;
  if (!data) return null;

  return (
    <div className="space-y-4 text-sm">
      <p className="text-ink-2">
        Baseline:{" "}
        <Link href={`/experiments/${data.baseline_experiment_id}`} className="text-accent-ink hover:underline">
          {data.baseline_experiment_id?.slice(0, 8)}…
        </Link>
        {" · "}threshold {data.threshold_pp} pp{" · "}prompt {data.prompt_changed ? "changed" : "unchanged"}
        {active && " · runs still in progress, verdict may change"}
      </p>
      {data.reasons.length > 0 && (
        <div className="rounded-md bg-neutral-bg p-3">
          <p className="font-medium">No verdict: this comparison is not valid</p>
          <ul className="mt-1 list-disc pl-5 text-ink-2">
            {data.reasons.map((reason) => <li key={reason} className="break-words">{reason}</li>)}
          </ul>
        </div>
      )}
      {data.models.map((model) => (
        <div key={`${model.provider}/${model.model_id}`} className="rounded-md border border-line p-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium">{modelLabel(model.provider, model.model_id)}</span>
            <VerdictBadge verdict={model.verdict} />
          </div>
          {model.reasons.length > 0 && (
            <ul className="mt-2 list-disc pl-5 text-ink-2">
              {model.reasons.map((reason) => <li key={reason} className="break-words">{reason}</li>)}
            </ul>
          )}
          {model.metrics.length > 0 && (
            <div className="mt-3 overflow-x-auto">
              <table className="w-full min-w-[560px] text-left">
                <thead className="text-xs uppercase tracking-wide text-ink-3">
                  <tr>
                    <th scope="col" className="py-1 pr-4 font-medium">Metric</th>
                    <th scope="col" className="py-1 pr-4 font-medium">Baseline</th>
                    <th scope="col" className="py-1 pr-4 font-medium">Candidate</th>
                    <th scope="col" className="py-1 pr-4 font-medium">Change</th>
                    <th scope="col" className="py-1 pr-4 font-medium">Paired</th>
                    <th scope="col" className="py-1 font-medium">Verdict</th>
                  </tr>
                </thead>
                <tbody className="tabular-nums">
                  {model.metrics.map((metric) => (
                    <tr key={metric.metric} className="border-t border-line">
                      <th scope="row" className="py-1.5 pr-4 text-left font-normal capitalize">{metric.metric}</th>
                      <td className="py-1.5 pr-4">{score(metric.baseline_mean)}</td>
                      <td className="py-1.5 pr-4">{score(metric.candidate_mean)}</td>
                      <td className="py-1.5 pr-4">{pp(metric.delta_pp)}</td>
                      <td className="py-1.5 pr-4">{metric.paired_cases}/{metric.applicable_cases}</td>
                      <td className="py-1.5">
                        <VerdictBadge verdict={metric.verdict} />
                      </td>
                    </tr>
                  ))}
                  {model.informational.map((info) => (
                    <tr key={info.name} className="border-t border-line text-ink-2">
                      <th scope="row" className="py-1.5 pr-4 text-left font-normal">{info.name.replaceAll("_", " ")}</th>
                      <td className="py-1.5 pr-4">{info.name.includes("cost") ? usd(info.baseline) : info.name.includes("latency") ? ms(info.baseline) : score(info.baseline)}</td>
                      <td className="py-1.5 pr-4">{info.name.includes("cost") ? usd(info.candidate) : info.name.includes("latency") ? ms(info.candidate) : score(info.candidate)}</td>
                      <td className="py-1.5 pr-4">{info.name.includes("facts") ? pp(info.delta) : "–"}</td>
                      <td className="py-1.5 pr-4">{info.paired_cases}</td>
                      <td className="py-1.5 text-xs text-ink-3">informational</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {model.metrics.some((m) => m.note) && (
            <ul className="mt-2 space-y-0.5 text-xs text-ink-3">
              {model.metrics.filter((m) => m.note).map((m) => (
                <li key={m.metric}><span className="capitalize">{m.metric}</span>: {m.note}</li>
              ))}
            </ul>
          )}
          {model.largest_drops.length > 0 && (
            <div className="mt-3 text-xs text-ink-2">
              Largest per-case drops:{" "}
              {model.largest_drops.slice(0, 5).map((drop, i) => (
                <span key={`${drop.case_id}-${drop.metric}`}>
                  {i > 0 && ", "}
                  <Link href={`/experiments/${experiment.id}/test/${encodeURIComponent(drop.case_id)}`} className="text-accent-ink hover:underline">
                    {drop.case_id}
                  </Link>{" "}
                  ({drop.metric} {pp(drop.delta_pp)})
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
      <p className="text-xs text-ink-3">{data.note}</p>
    </div>
  );
}

function flagTone(result: CaseOut["results"][number]): "critical" | "warning" | null {
  if (result.error) return "critical";
  if (result.flags.length) return "warning";
  return null;
}

function CaseTable({ experiment }: { experiment: ExperimentOut }) {
  const active = isActive(experiment);
  const [flaggedOnly, setFlaggedOnly] = useState(false);
  const { data, error, loading, reload } = useApi(
    () => api.results(experiment.id),
    [experiment.id, active],
    { pollMs: POLL_MS, shouldPoll: () => active },
  );
  if (loading && !data) return <Loading />;
  if (error && !data) return <ErrorState error={error} onRetry={reload} />;
  if (!data) return null;

  const runs = experiment.runs;
  const cases = flaggedOnly ? data.cases.filter((c) => c.results.some((r) => r.flags.length)) : data.cases;
  return (
    <>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-xs text-ink-2">
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={flaggedOnly} onChange={(e) => setFlaggedOnly(e.target.checked)} className="accent-[var(--accent)]" />
          Only cases with a flag (errors, missing facts, score below {data.low_score_threshold})
        </label>
        <span>Cells show correctness; open a case to see the response, judge reason and errors.</span>
      </div>
      {cases.length === 0 ? (
        <Empty title={flaggedOnly ? "No flagged cases" : "No cases"} />
      ) : (
        <div className="-mx-4 overflow-x-auto">
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead className="bg-sunken text-xs uppercase tracking-wide text-ink-3">
              <tr>
                <th scope="col" className="px-4 py-2 font-medium">Case</th>
                {runs.map((run) => (
                  <th key={run.id} scope="col" className="px-4 py-2 font-medium normal-case">{run.model_id}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {cases.map((c) => (
                <tr key={c.case_id} className="hover:bg-sunken/60">
                  <th scope="row" className="max-w-md px-4 py-2 text-left font-normal">
                    <Link href={`/experiments/${experiment.id}/test/${encodeURIComponent(c.case_id)}`} className="font-medium text-accent-ink hover:underline">
                      {c.case_id}
                    </Link>
                    <p className="truncate text-xs text-ink-3" title={c.input}>{c.input}</p>
                  </th>
                  {runs.map((run) => {
                    const result = c.results.find((r) => r.run_id === run.id);
                    if (!result) return <td key={run.id} className="px-4 py-2 text-xs text-ink-3">{active ? "pending…" : "no result"}</td>;
                    const tone = flagTone(result);
                    return (
                      <td key={run.id} className="px-4 py-2 tabular-nums">
                        {result.error ? (
                          <Badge tone="critical" title={result.error}>error</Badge>
                        ) : result.judge_error ? (
                          <Badge tone="warning" title={result.judge_error}>judge error</Badge>
                        ) : (
                          <span className={tone === "warning" ? "font-medium text-warning" : undefined}>
                            {score(result.correctness_score)}
                            {tone === "warning" && <span className="ml-1" title={result.flags.join(", ")}>!</span>}
                          </span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

function Progress({ experiment }: { experiment: ExperimentOut }) {
  const total = experiment.case_count * experiment.runs.length;
  const done = experiment.runs.reduce((sum, run: RunOut) => sum + (run.aggregate?.total_cases ?? 0), 0);
  return (
    <div role="status" className="mb-6 rounded-lg border border-line bg-raised p-4 text-sm">
      <div className="flex justify-between gap-2">
        <span>◷ Running… results appear as each case is scored.</span>
        <span className="tabular-nums text-ink-2">{done} / {total}</span>
      </div>
      <div className="mt-2 h-2 rounded-full bg-sunken" aria-hidden="true">
        <div className="h-2 rounded-full bg-accent transition-all" style={{ width: `${total ? (100 * done) / total : 0}%` }} />
      </div>
    </div>
  );
}

export default function ExperimentPage() {
  const { id } = useParams<{ id: string }>();
  const { data: experiment, error, loading, reload } = useApi(() => api.experiment(id), [id], {
    pollMs: POLL_MS,
    shouldPoll: isActive,
  });

  if (loading && !experiment) return <Loading />;
  if (error && !experiment) return <ErrorState error={error} onRetry={reload} />;
  if (!experiment) return null;

  const sample = experiment.runs.some((r) => r.sample);
  return (
    <>
      <PageHeader
        back={{ href: "/", label: "Dashboard" }}
        title={experiment.name}
        subtitle={
          <span className="flex flex-wrap items-center gap-2">
            <StatusBadge status={experiment.status} />
            {sample && <SampleBadge />}
            <span>{experiment.case_count} cases · created {dateTime(experiment.created_at)}</span>
          </span>
        }
        actions={<ButtonLink href={`/experiments/new?baseline=${experiment.id}`} variant="secondary">New candidate from this</ButtonLink>}
      />
      {sample && (
        <p className="mb-6 rounded-md border border-dashed border-warning bg-warning-bg px-3 py-2 text-xs text-ink">
          <strong>Sample mode:</strong> sample runs use offline heuristic models and a heuristic judge with simulated latency and fictional prices. They show the pipeline working, not real model quality.
        </p>
      )}
      {isActive(experiment) && <Progress experiment={experiment} />}

      <div className="space-y-6">
        <details className="rounded-lg border border-line bg-raised px-4 py-3 text-sm">
          <summary className="cursor-pointer font-medium">System prompt</summary>
          <pre className="mt-3 whitespace-pre-wrap break-words font-mono text-xs text-ink-2">{experiment.system_prompt}</pre>
        </details>

        <Card title="Model comparison">
          <ModelTable experiment={experiment} />
        </Card>

        <Card title="Regression vs. baseline">
          <RegressionPanel experiment={experiment} />
        </Card>

        <Card title="Quality vs. cost">
          <div className="max-w-3xl">
            <QualityCostChart runs={experiment.runs} />
          </div>
        </Card>

        <Card title="Per-case results">
          <CaseTable experiment={experiment} />
        </Card>
      </div>
    </>
  );
}
