"use client";

import Link from "next/link";
import { useParams } from "next/navigation";

import { Badge, Card, ErrorState, Loading, PageHeader, SampleBadge, StatusBadge } from "@/components/ui";
import { api, type CaseResult } from "@/lib/api";
import { modelLabel, ms, score, tokens, usd } from "@/lib/format";
import { useApi } from "@/lib/useApi";

const FLAG_LABELS: Record<string, string> = {
  candidate_error: "request error",
  judge_error: "judge error",
  missing_facts: "missing facts",
  low_correctness: "low correctness",
  low_groundedness: "low groundedness",
};

function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-md bg-sunken px-3 py-2">
      <dt className="text-[11px] uppercase tracking-wide text-ink-3">{label}</dt>
      <dd className="mt-0.5 font-medium tabular-nums">{value}</dd>
      {hint && <dd className="text-[11px] text-ink-3">{hint}</dd>}
    </div>
  );
}

function ResultCard({ result, requiredFacts, hasContext }: { result: CaseResult; requiredFacts: string[]; hasContext: boolean }) {
  return (
    <Card
      title={
        <span className="flex flex-wrap items-center gap-2">
          {modelLabel(result.provider, result.model_id)}
          {result.sample && <SampleBadge />}
          <StatusBadge status={result.run_status} />
        </span>
      }
      actions={
        <div className="flex flex-wrap gap-1">
          {result.flags.map((flag) => (
            <Badge key={flag} tone={flag === "candidate_error" ? "critical" : "warning"}>{FLAG_LABELS[flag] ?? flag}</Badge>
          ))}
        </div>
      }
    >
      <div className="space-y-4 text-sm">
        {result.error && (
          <div role="alert" className="rounded-md bg-critical-bg p-3">
            <p className="font-medium text-critical">✕ The request failed; nothing was scored</p>
            <p className="mt-1 break-words font-mono text-xs text-ink">{result.error}</p>
          </div>
        )}
        {result.response != null && (
          <div>
            <h3 className="text-xs font-medium uppercase tracking-wide text-ink-3">Response</h3>
            <p className="mt-1 whitespace-pre-wrap break-words rounded-md border border-line bg-surface p-3">{result.response}</p>
          </div>
        )}

        <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Metric label="Correctness" value={score(result.correctness_score)} hint="judge, 0–100" />
          <Metric label="Groundedness" value={hasContext ? score(result.groundedness_score) : "N/A"} hint={hasContext ? "judge, 0–100" : "no context"} />
          <Metric
            label="Required facts"
            value={requiredFacts.length ? score(result.required_facts_score) : "N/A"}
            hint={requiredFacts.length ? "lexical match" : "none listed"}
          />
          <Metric label="Latency" value={ms(result.latency_ms)} hint="final attempt" />
          <Metric label="Tokens in / out" value={`${tokens(result.input_tokens)} / ${tokens(result.output_tokens)}`} hint="provider-reported" />
          <Metric label="Candidate cost" value={usd(result.estimated_cost)} />
          <Metric label="Judge tokens" value={`${tokens(result.judge_input_tokens)} / ${tokens(result.judge_output_tokens)}`} />
          <Metric label="Judge cost" value={usd(result.judge_cost)} />
        </dl>

        {requiredFacts.length > 0 && result.response != null && (
          <div>
            <h3 className="text-xs font-medium uppercase tracking-wide text-ink-3">Required facts in this response</h3>
            <ul className="mt-1 flex flex-wrap gap-1.5">
              {requiredFacts.map((fact) => {
                const missing = result.missing_facts.includes(fact);
                return (
                  <li key={fact}>
                    <Badge tone={missing ? "critical" : "good"}>{missing ? `missing: ${fact}` : fact}</Badge>
                  </li>
                );
              })}
            </ul>
          </div>
        )}

        {(result.judge_reason || result.judge_error) && (
          <div>
            <h3 className="text-xs font-medium uppercase tracking-wide text-ink-3">
              Judge ({result.judge_provider}/{result.judge_model_id})
            </h3>
            {result.judge_error ? (
              <p className="mt-1 rounded-md bg-warning-bg p-3 text-xs">
                <span className="font-medium text-warning">! Scoring failed:</span> <span className="break-words">{result.judge_error}</span>
              </p>
            ) : (
              <p className="mt-1 text-ink-2">{result.judge_reason}</p>
            )}
            <p className="mt-1 text-xs text-ink-3">Judge scores are estimates from a model, not verified facts.</p>
          </div>
        )}
      </div>
    </Card>
  );
}

export default function CaseInspector() {
  const { id, testId } = useParams<{ id: string; testId: string }>();
  const caseId = decodeURIComponent(testId);
  const { data, error, loading, reload } = useApi(() => api.results(id), [id]);

  if (loading && !data) return <Loading />;
  if (error && !data) return <ErrorState error={error} onRetry={reload} />;
  if (!data) return null;

  const index = data.cases.findIndex((c) => c.case_id === caseId);
  if (index === -1) return <ErrorState error={new Error(`Test case "${caseId}" is not part of this experiment.`)} />;
  const testCase = data.cases[index];
  const prev = data.cases[index - 1];
  const next = data.cases[index + 1];
  const caseHref = (c: { case_id: string }) => `/experiments/${id}/test/${encodeURIComponent(c.case_id)}`;

  return (
    <>
      <PageHeader
        back={{ href: `/experiments/${id}`, label: "Experiment results" }}
        title={<span className="font-mono">{testCase.case_id}</span>}
        subtitle={`Case ${index + 1} of ${data.cases.length}`}
        actions={
          <nav aria-label="Case navigation" className="flex gap-2 text-sm">
            {prev ? <Link className="rounded-md border border-line bg-raised px-3 py-1.5 hover:bg-sunken" href={caseHref(prev)}>← {prev.case_id}</Link> : null}
            {next ? <Link className="rounded-md border border-line bg-raised px-3 py-1.5 hover:bg-sunken" href={caseHref(next)}>{next.case_id} →</Link> : null}
          </nav>
        }
      />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <div className="space-y-4 lg:sticky lg:top-6 lg:self-start">
          <Card title="Question">
            <p className="whitespace-pre-wrap break-words text-sm">{testCase.input}</p>
          </Card>
          <Card title="Context">
            {testCase.context ? (
              <p className="whitespace-pre-wrap break-words text-sm text-ink-2">{testCase.context}</p>
            ) : (
              <p className="text-sm text-ink-3">No context supplied: groundedness is N/A for this case.</p>
            )}
          </Card>
          <Card title="Expected answer">
            <p className="whitespace-pre-wrap break-words text-sm">{testCase.expected_answer}</p>
            <div className="mt-3 text-xs text-ink-3">
              Required facts:{" "}
              {testCase.required_facts.length ? testCase.required_facts.map((f) => `“${f}”`).join(", ") : "none (coverage is N/A)"}
            </div>
            {Object.keys(testCase.metadata).length > 0 && (
              <dl className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-3">
                {Object.entries(testCase.metadata).map(([key, value]) => (
                  <div key={key}><dt className="inline">{key}:</dt> <dd className="inline text-ink-2">{String(value)}</dd></div>
                ))}
              </dl>
            )}
          </Card>
        </div>
        <div className="space-y-4">
          {testCase.results.length === 0 ? (
            <Card><p className="text-sm text-ink-2">No model has produced a result for this case yet.</p></Card>
          ) : (
            testCase.results.map((result) => (
              <ResultCard key={result.run_id} result={result} requiredFacts={testCase.required_facts} hasContext={testCase.context != null} />
            ))
          )}
        </div>
      </div>
    </>
  );
}
