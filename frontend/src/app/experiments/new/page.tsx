"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";

import { Badge, Card, ErrorState, Loading, PageHeader, SampleBadge } from "@/components/ui";
import { ApiError, api, type DatasetValidation, type ModelInfo, type TestCaseIn } from "@/lib/api";
import { dateTime, modelLabel } from "@/lib/format";
import { useApi } from "@/lib/useApi";

const MAX_PROMPT_CHARS = 8000;
const DEFAULT_PROMPT =
  "You are a customer-support assistant for Northwind Outfitters. Answer using only the policy context provided. " +
  "If the context does not answer the question, say so and suggest contacting support.";

type DatasetSource = { label: string; cases: TestCaseIn[]; raw: unknown };

const inputClass =
  "w-full rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-3 focus:border-accent";

function modelKey(model: Pick<ModelInfo, "provider" | "model_id">): string {
  return `${model.provider}/${model.model_id}`;
}

function Field({ label, htmlFor, hint, children }: { label: string; htmlFor?: string; hint?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={htmlFor} className="block text-sm font-medium">
        {label}
      </label>
      {children}
      {hint && <div className="text-xs text-ink-3">{hint}</div>}
    </div>
  );
}

function NewExperimentForm() {
  const router = useRouter();
  const params = useSearchParams();
  const baselineParam = params.get("baseline");

  const models = useApi(() => api.models(), []);
  const experiments = useApi(() => api.experiments(), []);

  const [name, setName] = useState("");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [dataset, setDataset] = useState<DatasetSource | null>(null);
  const [validation, setValidation] = useState<DatasetValidation | null>(null);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [baselineId, setBaselineId] = useState<string>("");
  const [threshold, setThreshold] = useState("3");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<{ messages: string[]; experimentId?: string } | null>(null);

  async function applyDataset(label: string, raw: unknown) {
    setDatasetError(null);
    setValidation(null);
    try {
      const result = await api.validateDataset(raw);
      setValidation(result);
      const cases = (Array.isArray(raw) ? raw : (raw as { cases?: TestCaseIn[] })?.cases) ?? [];
      setDataset({ label, cases: cases as TestCaseIn[], raw });
    } catch (err) {
      setDataset(null);
      setDatasetError(err instanceof Error ? err.message : String(err));
    }
  }

  async function loadSample() {
    try {
      const sample = await api.sampleDataset();
      await applyDataset(`Sample dataset (${sample.name ?? "customer_support"})`, sample);
    } catch (err) {
      setDatasetError(err instanceof Error ? err.message : String(err));
    }
  }

  async function onFile(file: File | undefined) {
    if (!file) return;
    let raw: unknown;
    try {
      raw = JSON.parse(await file.text());
    } catch {
      setDataset(null);
      setValidation(null);
      setDatasetError(`${file.name} is not valid JSON.`);
      return;
    }
    await applyDataset(file.name, raw);
  }

  // "New candidate from this baseline": reuse the baseline's exact cases and prompt.
  useEffect(() => {
    if (!baselineParam) return;
    let cancelled = false;
    (async () => {
      try {
        const [experiment, results] = await Promise.all([api.experiment(baselineParam), api.results(baselineParam)]);
        if (cancelled) return;
        const cases: TestCaseIn[] = results.cases.map((c) => ({
          id: c.case_id,
          input: c.input,
          context: c.context,
          expected_answer: c.expected_answer,
          required_facts: c.required_facts,
          metadata: c.metadata,
        }));
        setName(`${experiment.name} (candidate)`);
        setPrompt(experiment.system_prompt);
        setBaselineId(experiment.id);
        setThreshold(String(experiment.regression_threshold));
        setSelected(new Set(experiment.runs.map((r) => modelKey(r))));
        await applyDataset(`Cases from baseline "${experiment.name}"`, { cases });
      } catch (err) {
        if (!cancelled) setDatasetError(`Could not load baseline: ${err instanceof Error ? err.message : err}`);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [baselineParam]);

  const compatibleBaselines = useMemo(
    () => (experiments.data ?? []).filter((e) => validation?.dataset_hash && e.dataset_hash === validation.dataset_hash),
    [experiments.data, validation],
  );
  const incompatibleCount = (experiments.data?.length ?? 0) - compatibleBaselines.length;

  // A baseline chosen earlier stops applying if the dataset changes to different cases.
  const effectiveBaselineId = compatibleBaselines.some((e) => e.id === baselineId) ? baselineId : "";

  const availableModels = models.data?.models ?? [];
  const selectedModels = availableModels.filter((m) => m.available && selected.has(modelKey(m)));
  const needsJudge = selectedModels.some((m) => !m.sample);
  const judgeBlocked = needsJudge && !models.data?.judge.available;
  const thresholdValue = Number(threshold);
  const thresholdValid = Number.isFinite(thresholdValue) && thresholdValue > 0 && thresholdValue <= 100;

  const problems: string[] = [];
  if (!name.trim()) problems.push("Give the experiment a name.");
  if (!prompt.trim()) problems.push("The system prompt is empty.");
  if (prompt.length > MAX_PROMPT_CHARS) problems.push(`The system prompt is over ${MAX_PROMPT_CHARS.toLocaleString()} characters.`);
  if (!validation?.valid) problems.push("Load a valid dataset.");
  if (selectedModels.length === 0) problems.push("Select at least one available model.");
  if (judgeBlocked) problems.push(`Real models need a judge: ${models.data?.judge.reason}.`);
  if (!thresholdValid) problems.push("The regression threshold must be between 0 and 100 percentage points.");

  function toggle(model: ModelInfo) {
    setSelected((current) => {
      const next = new Set(current);
      const key = modelKey(model);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (problems.length || !dataset) return;
    setSubmitting(true);
    setSubmitError(null);
    let experimentId: string | undefined;
    try {
      const experiment = await api.createExperiment({
        name: name.trim(),
        system_prompt: prompt,
        cases: dataset.cases,
        baseline_experiment_id: effectiveBaselineId || null,
        regression_threshold: thresholdValue,
      });
      experimentId = experiment.id;
      await api.run(experiment.id, selectedModels.map(({ provider, model_id }) => ({ provider, model_id })));
      router.push(`/experiments/${experiment.id}`);
    } catch (err) {
      setSubmitting(false);
      setSubmitError({
        messages: err instanceof ApiError ? err.details : [String(err)],
        experimentId,
      });
    }
  }

  if (models.loading || experiments.loading) return <Loading />;
  if (models.error) return <ErrorState error={models.error} onRetry={models.reload} />;

  const judge = models.data!.judge;
  return (
    <form onSubmit={submit} className="grid gap-6 lg:grid-cols-[1fr_320px]" noValidate>
      <div className="space-y-6">
        <Card title="1. Experiment">
          <div className="space-y-4">
            <Field label="Name" htmlFor="name">
              <input id="name" className={inputClass} value={name} onChange={(e) => setName(e.target.value)} maxLength={200} placeholder="e.g. Support bot – prompt v2" required />
            </Field>
            <Field
              label="System prompt"
              htmlFor="prompt"
              hint={
                <span className={prompt.length > MAX_PROMPT_CHARS ? "text-critical" : undefined}>
                  {prompt.length.toLocaleString()} / {MAX_PROMPT_CHARS.toLocaleString()} characters. Sent as the system message; each case&apos;s context and question form the user message.
                </span>
              }
            >
              <textarea id="prompt" className={`${inputClass} min-h-32 font-mono`} value={prompt} onChange={(e) => setPrompt(e.target.value)} required />
            </Field>
          </div>
        </Card>

        <Card title="2. Dataset">
          <div className="flex flex-wrap items-center gap-3">
            <button type="button" onClick={loadSample} className="rounded-md border border-line bg-raised px-3 py-2 text-sm hover:bg-sunken">
              Use sample dataset
            </button>
            <label className="cursor-pointer rounded-md border border-line bg-raised px-3 py-2 text-sm hover:bg-sunken">
              Upload JSON…
              <input type="file" accept="application/json,.json" className="sr-only" onChange={(e) => onFile(e.target.files?.[0])} />
            </label>
            <span className="text-xs text-ink-3">
              Format: <code>{"{ \"cases\": [ { id, input, context?, expected_answer, required_facts?, metadata? } ] }"}</code> or a bare array. Up to 200 cases.
            </span>
          </div>
          <div aria-live="polite" className="mt-4 text-sm">
            {datasetError && <p className="text-critical">✕ {datasetError}</p>}
            {validation?.valid && dataset && (
              <p className="text-good">
                ✓ {dataset.label}: {validation.case_count} valid cases
                {" "}
                <span className="text-ink-3">(content hash {validation.dataset_hash?.slice(0, 10)}…)</span>
              </p>
            )}
            {validation && !validation.valid && (
              <div className="text-critical">
                <p>✕ {dataset?.label ?? "Dataset"} failed validation:</p>
                <ul className="mt-1 list-disc space-y-0.5 pl-5 text-ink">
                  {validation.errors.slice(0, 20).map((error) => (
                    <li key={error} className="break-words">{error}</li>
                  ))}
                  {validation.errors.length > 20 && <li>…and {validation.errors.length - 20} more</li>}
                </ul>
              </div>
            )}
          </div>
        </Card>

        <Card title="3. Models">
          {availableModels.length === 0 ? (
            <p className="text-sm text-ink-2">
              No models are configured. Add model IDs and API keys to <code>.env</code> (see README), or set <code>SAMPLE_MODE=true</code>.
            </p>
          ) : (
            <fieldset className="space-y-2">
              <legend className="sr-only">Models to evaluate</legend>
              {availableModels.map((model) => {
                const key = modelKey(model);
                return (
                  <label
                    key={key}
                    className={`flex items-start gap-3 rounded-md border border-line p-3 text-sm ${model.available ? "cursor-pointer hover:bg-sunken" : "opacity-60"}`}
                  >
                    <input
                      type="checkbox"
                      className="mt-0.5 h-4 w-4 accent-[var(--accent)]"
                      checked={selected.has(key) && model.available}
                      disabled={!model.available}
                      onChange={() => toggle(model)}
                    />
                    <span className="min-w-0 flex-1">
                      <span className="flex flex-wrap items-center gap-2 font-medium">
                        {modelLabel(model.provider, model.model_id)}
                        {model.sample && <SampleBadge />}
                      </span>
                      {!model.available && <span className="mt-0.5 block text-xs text-ink-3">Unavailable: {model.reason}</span>}
                    </span>
                  </label>
                );
              })}
            </fieldset>
          )}
          <div className="mt-4 rounded-md bg-sunken p-3 text-xs text-ink-2">
            <p>
              <span className="font-medium text-ink">Judge:</span>{" "}
              {judge.provider && judge.model_id ? modelLabel(judge.provider, judge.model_id) : "not configured"}{" "}
              {judge.available ? <Badge tone="good">available</Badge> : <Badge tone="warning">{judge.reason ?? "unavailable"}</Badge>}
            </p>
            <p className="mt-1">Sample models are always scored by the offline heuristic sample judge, never by a real LLM.</p>
          </div>
        </Card>

        <Card title="4. Baseline (optional)">
          <div className="grid gap-4 sm:grid-cols-[1fr_160px]">
            <Field
              label="Compare against"
              htmlFor="baseline"
              hint={
                validation?.valid
                  ? `${compatibleBaselines.length} experiment(s) use exactly these cases${incompatibleCount > 0 ? `; ${incompatibleCount} other(s) hidden because their cases differ` : ""}.`
                  : "Load a dataset first; only experiments with identical cases can be baselines."
              }
            >
              <select id="baseline" className={inputClass} value={effectiveBaselineId} onChange={(e) => setBaselineId(e.target.value)} disabled={!validation?.valid}>
                <option value="">No baseline</option>
                {compatibleBaselines.map((e) => (
                  <option key={e.id} value={e.id}>
                    {e.name} · {e.models.join(", ") || "not run"} · {dateTime(e.created_at)}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Threshold (pp)" htmlFor="threshold" hint="Flag drops larger than this.">
              <input id="threshold" type="number" min={0.1} max={100} step={0.5} className={inputClass} value={threshold} onChange={(e) => setThreshold(e.target.value)} />
            </Field>
          </div>
        </Card>
      </div>

      <aside className="lg:sticky lg:top-6 lg:self-start">
        <Card title="Run">
          <dl className="space-y-1 text-sm">
            <div className="flex justify-between gap-2"><dt className="text-ink-2">Cases</dt><dd className="tabular-nums">{validation?.valid ? validation.case_count : "–"}</dd></div>
            <div className="flex justify-between gap-2"><dt className="text-ink-2">Models</dt><dd className="tabular-nums">{selectedModels.length}</dd></div>
            <div className="flex justify-between gap-2">
              <dt className="text-ink-2">Model calls</dt>
              <dd className="tabular-nums">{validation?.valid ? (validation.case_count ?? 0) * selectedModels.length * 2 : "–"}</dd>
            </div>
          </dl>
          <p className="mt-2 text-xs text-ink-3">One candidate call plus one judge call per case per model, plus bounded retries. Real providers bill for these.</p>
          {problems.length > 0 && (
            <ul className="mt-4 space-y-1 text-xs text-ink-2" aria-live="polite">
              {problems.map((p) => (
                <li key={p}>• {p}</li>
              ))}
            </ul>
          )}
          {submitError && (
            <div role="alert" className="mt-4 rounded-md bg-critical-bg p-3 text-xs">
              <p className="font-medium text-critical">✕ {submitError.experimentId ? "Experiment created, but the run was rejected:" : "Could not create the experiment:"}</p>
              <ul className="mt-1 list-disc pl-4 text-ink">
                {submitError.messages.map((m) => (
                  <li key={m} className="break-words">{m}</li>
                ))}
              </ul>
              {submitError.experimentId && (
                <a className="mt-2 inline-block text-accent-ink underline" href={`/experiments/${submitError.experimentId}`}>
                  Open the experiment
                </a>
              )}
            </div>
          )}
          <button
            type="submit"
            disabled={problems.length > 0 || submitting}
            className="mt-4 w-full rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-contrast hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {submitting ? "Starting…" : "Create and run"}
          </button>
        </Card>
      </aside>
    </form>
  );
}

export default function NewExperimentPage() {
  return (
    <>
      <PageHeader title="New evaluation" subtitle="Pick a prompt, a dataset and models. Runs execute on the backend; you can watch progress on the results page." back={{ href: "/", label: "Dashboard" }} />
      <Suspense fallback={<Loading />}>
        <NewExperimentForm />
      </Suspense>
    </>
  );
}
