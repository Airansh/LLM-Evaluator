"use client";

import Link from "next/link";

import { ButtonLink, Card, Empty, ErrorState, Loading, PageHeader, RegressionBadge, SampleBadge, StatusBadge } from "@/components/ui";
import { api, type ExperimentSummary } from "@/lib/api";
import { dateTime } from "@/lib/format";
import { useApi } from "@/lib/useApi";

function Stat({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <div className="rounded-lg border border-line bg-raised p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-ink-3">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums">{value}</p>
      {hint && <p className="mt-0.5 text-xs text-ink-3">{hint}</p>}
    </div>
  );
}

function Summary({ experiments }: { experiments: ExperimentSummary[] }) {
  const evaluations = experiments.reduce((sum, e) => sum + e.case_count * e.run_count, 0);
  const running = experiments.filter((e) => e.status === "running").length;
  const regressions = experiments.filter((e) => e.regression === "regression").length;
  return (
    <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4">
      <Stat label="Experiments" value={experiments.length} />
      <Stat label="Case evaluations" value={evaluations.toLocaleString("en-US")} hint="cases × model runs" />
      <Stat label="Running" value={running} />
      <Stat label="Regressions flagged" value={regressions} hint="vs. each experiment's baseline" />
    </div>
  );
}

export default function Dashboard() {
  const { data, error, loading, reload } = useApi(() => api.experiments(), [], {
    pollMs: 3000,
    shouldPoll: (list) => list.some((e) => e.status === "running"),
  });

  return (
    <>
      <PageHeader
        title="Experiments"
        subtitle="Run one dataset against several models, then compare quality, latency and estimated cost against a baseline."
        actions={<ButtonLink href="/experiments/new">+ New evaluation</ButtonLink>}
      />
      {loading && !data && <Loading />}
      {error && !data ? <ErrorState error={error} onRetry={reload} /> : null}
      {data && data.length === 0 && (
        <Empty title="No experiments yet">
          <p>Create one from the bundled sample dataset or upload your own JSON.</p>
          <div className="mt-4">
            <ButtonLink href="/experiments/new">Start a new evaluation</ButtonLink>
          </div>
        </Empty>
      )}
      {data && data.length > 0 && (
        <>
          <Summary experiments={data} />
          <Card title="Recent experiments" className="overflow-hidden">
            <div className="-m-4 overflow-x-auto">
              <table className="w-full min-w-[720px] text-left text-sm">
                <thead className="bg-sunken text-xs uppercase tracking-wide text-ink-3">
                  <tr>
                    <th scope="col" className="px-4 py-2 font-medium">Name</th>
                    <th scope="col" className="px-4 py-2 font-medium">Status</th>
                    <th scope="col" className="px-4 py-2 font-medium">Models</th>
                    <th scope="col" className="px-4 py-2 text-right font-medium">Cases</th>
                    <th scope="col" className="px-4 py-2 font-medium">Regression</th>
                    <th scope="col" className="px-4 py-2 font-medium">Created</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {data.map((experiment) => (
                    <tr key={experiment.id} className="hover:bg-sunken/60">
                      <td className="px-4 py-3">
                        <Link href={`/experiments/${experiment.id}`} className="font-medium text-accent-ink hover:underline">
                          {experiment.name}
                        </Link>
                        {experiment.sample && <span className="ml-2"><SampleBadge /></span>}
                      </td>
                      <td className="px-4 py-3"><StatusBadge status={experiment.status} /></td>
                      <td className="px-4 py-3 text-ink-2">
                        {experiment.models.length ? experiment.models.join(", ") : <span className="text-ink-3">not run</span>}
                      </td>
                      <td className="px-4 py-3 text-right tabular-nums">{experiment.case_count}</td>
                      <td className="px-4 py-3"><RegressionBadge value={experiment.regression} /></td>
                      <td className="px-4 py-3 whitespace-nowrap text-ink-2">{dateTime(experiment.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </>
  );
}
