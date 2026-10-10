import Link from "next/link";
import type { ReactNode } from "react";

import type { ExperimentStatus, MetricVerdict, ModelVerdict, RegressionIndicator, RunStatus } from "@/lib/api";

type Tone = "good" | "warning" | "serious" | "critical" | "neutral" | "info";

const TONE_CLASSES: Record<Tone, string> = {
  good: "bg-good-bg text-good",
  warning: "bg-warning-bg text-warning",
  serious: "bg-serious-bg text-serious",
  critical: "bg-critical-bg text-critical",
  neutral: "bg-neutral-bg text-ink-2",
  info: "bg-neutral-bg text-accent-ink",
};

// Status never relies on color alone: every badge carries an icon glyph and a text label.
const TONE_ICONS: Record<Tone, string> = {
  good: "✓",
  warning: "!",
  serious: "!",
  critical: "✕",
  neutral: "•",
  info: "◷",
};

export function Badge({ tone, children, title }: { tone: Tone; children: ReactNode; title?: string }) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ${TONE_CLASSES[tone]}`}
    >
      <span aria-hidden="true">{TONE_ICONS[tone]}</span>
      {children}
    </span>
  );
}

const STATUS_TONES: Record<RunStatus | ExperimentStatus, Tone> = {
  draft: "neutral",
  pending: "info",
  running: "info",
  completed: "good",
  partial: "warning",
  failed: "critical",
};

export function StatusBadge({ status }: { status: RunStatus | ExperimentStatus }) {
  return <Badge tone={STATUS_TONES[status]}>{status}</Badge>;
}

export function RegressionBadge({ value }: { value: RegressionIndicator }) {
  if (value === null) return <span className="text-xs text-ink-3">no baseline</span>;
  if (value === "regression") return <Badge tone="critical">regression</Badge>;
  if (value === "ok") return <Badge tone="good">no regression</Badge>;
  return (
    <Badge tone="neutral" title="Not comparable or incomplete data; open the experiment for details">
      no verdict
    </Badge>
  );
}

const VERDICT_LABELS: Record<ModelVerdict | MetricVerdict, [Tone, string]> = {
  regression: ["critical", "regression"],
  ok: ["good", "within threshold"],
  insufficient_data: ["warning", "insufficient data"],
  not_comparable: ["neutral", "not comparable"],
  not_applicable: ["neutral", "N/A"],
};

export function VerdictBadge({ verdict }: { verdict: ModelVerdict | MetricVerdict }) {
  const [tone, label] = VERDICT_LABELS[verdict];
  return <Badge tone={tone}>{label}</Badge>;
}

export function SampleBadge() {
  return (
    <span
      title="Offline sample provider: heuristic output, simulated latency, fictional prices. Not a real LLM."
      className="inline-flex items-center rounded border border-dashed border-warning px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-warning"
    >
      sample
    </span>
  );
}

export function Card({ title, actions, children, className = "" }: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-lg border border-line bg-raised ${className}`}>
      {(title || actions) && (
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-3">
          {title && <h2 className="text-sm font-semibold">{title}</h2>}
          {actions}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function PageHeader({ title, subtitle, actions, back }: {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  back?: { href: string; label: string };
}) {
  return (
    <div className="mb-6 space-y-1">
      {back && (
        <Link href={back.href} className="text-sm text-accent-ink hover:underline">
          ← {back.label}
        </Link>
      )}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="break-words text-2xl font-semibold tracking-tight">{title}</h1>
          {subtitle && <div className="mt-1 text-sm text-ink-2">{subtitle}</div>}
        </div>
        {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
      </div>
    </div>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div role="status" className="flex items-center gap-2 py-10 text-sm text-ink-2">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-line border-t-accent" aria-hidden="true" />
      {label}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div role="alert" className="rounded-lg border border-critical/40 bg-critical-bg p-4 text-sm">
      <p className="font-medium text-critical">✕ Something went wrong</p>
      <p className="mt-1 break-words text-ink">{message}</p>
      {onRetry && (
        <button type="button" onClick={onRetry} className="mt-3 rounded-md border border-line bg-raised px-3 py-1.5 text-sm hover:bg-sunken">
          Try again
        </button>
      )}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-line px-6 py-10 text-center">
      <p className="font-medium">{title}</p>
      {children && <div className="mt-2 text-sm text-ink-2">{children}</div>}
    </div>
  );
}

export function ButtonLink({ href, children, variant = "primary" }: {
  href: string;
  children: ReactNode;
  variant?: "primary" | "secondary";
}) {
  const styles =
    variant === "primary"
      ? "bg-accent text-accent-contrast hover:opacity-90"
      : "border border-line bg-raised text-ink hover:bg-sunken";
  return (
    <Link href={href} className={`inline-flex items-center rounded-md px-3.5 py-2 text-sm font-medium ${styles}`}>
      {children}
    </Link>
  );
}
