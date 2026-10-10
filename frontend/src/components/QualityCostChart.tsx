"use client";

import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";

import type { RunOut } from "@/lib/api";
import { score, usd } from "@/lib/format";

interface Point {
  label: string;
  cost: number;
  correctness: number;
  n: number;
  nPriced: number;
  total: number;
  sample: boolean;
  labelLeft: boolean;
}

// One point per model run. A single hue plus a direct text label carries identity (no
// color-only legend); sample runs use a hollow marker as secondary encoding.
function Marker(props: { cx?: number; cy?: number; payload?: Point }) {
  const { cx = 0, cy = 0, payload } = props;
  if (!payload) return null;
  return (
    <g>
      <circle
        cx={cx}
        cy={cy}
        r={6}
        style={{
          fill: payload.sample ? "var(--surface-raised)" : "var(--series-1)",
          stroke: payload.sample ? "var(--series-1)" : "var(--surface-raised)",
          strokeWidth: 2,
        }}
      />
      <text
        x={payload.labelLeft ? cx - 10 : cx + 10}
        y={cy + 4}
        textAnchor={payload.labelLeft ? "end" : "start"}
        style={{ fill: "var(--text-secondary)", fontSize: 12 }}
      >
        {payload.label}
      </text>
    </g>
  );
}

function ChartTooltip({ active, payload }: { active?: boolean; payload?: { payload: Point }[] }) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <div className="rounded-md border border-line bg-raised px-3 py-2 text-xs shadow-sm">
      <p className="font-medium text-ink">{point.label}{point.sample ? " (sample)" : ""}</p>
      <p className="text-ink-2">Mean correctness: {score(point.correctness)} (n={point.n})</p>
      <p className="text-ink-2">
        Candidate cost: {usd(point.cost)} ({point.nPriced}/{point.total} cases priced)
      </p>
    </div>
  );
}

export function QualityCostChart({ runs }: { runs: RunOut[] }) {
  const points: Point[] = [];
  const excluded: { label: string; reason: string }[] = [];
  for (const run of runs) {
    const agg = run.aggregate;
    const label = run.model_id;
    if (!agg) continue;
    if (agg.correctness.mean == null) excluded.push({ label, reason: "no scored cases" });
    else if (agg.candidate_cost.total == null) excluded.push({ label, reason: "cost unavailable (no price or usage)" });
    else
      points.push({
        label,
        cost: agg.candidate_cost.total,
        correctness: agg.correctness.mean,
        n: agg.correctness.n,
        nPriced: agg.candidate_cost.n_priced,
        total: agg.total_cases,
        sample: run.sample,
        labelLeft: false,
      });
  }
  // Avoid label collisions: alternate label sides by cost order, so neighbours face away from each other.
  [...points].sort((a, b) => a.cost - b.cost).forEach((point, i, sorted) => {
    point.labelLeft = sorted.length > 1 && i % 2 === 0;
  });

  const maxCost = Math.max(...points.map((p) => p.cost), 0);
  return (
    <figure>
      {points.length === 0 ? (
        <p className="py-8 text-center text-sm text-ink-2">
          Nothing to plot yet: a run needs both scored cases and a measurable cost.
        </p>
      ) : (
        <div className="h-72 w-full" role="img" aria-label="Scatter plot of mean correctness versus total estimated candidate cost, one point per model run. The same values are in the model comparison table.">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 12, right: 110, bottom: 28, left: 4 }}>
              <CartesianGrid stroke="var(--grid)" strokeDasharray="0" />
              <XAxis
                type="number"
                dataKey="cost"
                name="Candidate cost"
                domain={[0, maxCost > 0 ? maxCost * 1.15 : 1]}
                tickFormatter={(v: number) => usd(v)}
                tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                stroke="var(--border)"
                label={{ value: "Total estimated candidate cost (USD)", position: "insideBottom", offset: -16, fill: "var(--text-muted)", fontSize: 12 }}
              />
              <YAxis
                type="number"
                dataKey="correctness"
                name="Mean correctness"
                domain={[0, 100]}
                ticks={[0, 25, 50, 75, 100]}
                tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                stroke="var(--border)"
                width={40}
                label={{ value: "Mean correctness", angle: -90, position: "insideLeft", offset: 12, fill: "var(--text-muted)", fontSize: 12, dy: 50 }}
              />
              <Tooltip content={<ChartTooltip />} cursor={{ stroke: "var(--border)" }} />
              <Scatter data={points} shape={<Marker />} isAnimationActive={false} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
      <figcaption className="mt-2 space-y-1 text-xs text-ink-3">
        <p>Higher and further left is better. Correctness is a judge estimate over successfully scored cases; costs exclude judge (evaluation) cost. Hollow markers are sample-mode runs.</p>
        {excluded.length > 0 && (
          <p>Not plotted: {excluded.map((e) => `${e.label} (${e.reason})`).join("; ")}.</p>
        )}
      </figcaption>
    </figure>
  );
}
