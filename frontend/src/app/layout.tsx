import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";

import "./globals.css";

export const metadata: Metadata = {
  title: "EvalForge",
  description: "Run one evaluation dataset against multiple LLMs, compare quality, latency and cost, and catch regressions.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="flex min-h-full flex-col">
        <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-2 focus:z-10 focus:rounded focus:bg-raised focus:px-3 focus:py-2">
          Skip to content
        </a>
        <header className="border-b border-line bg-raised">
          <nav aria-label="Main" className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3">
            <Link href="/" className="font-semibold tracking-tight">
              EvalForge
            </Link>
            <div className="flex items-center gap-4 text-sm">
              <Link href="/" className="text-ink-2 hover:text-ink">
                Dashboard
              </Link>
              <Link href="/experiments/new" className="rounded-md bg-accent px-3 py-1.5 font-medium text-accent-contrast hover:opacity-90">
                New evaluation
              </Link>
            </div>
          </nav>
        </header>
        <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">
          {children}
        </main>
        <footer className="border-t border-line px-4 py-4 text-center text-xs text-ink-3">
          Judge scores are model estimates, not verified facts. Costs are estimates from a versioned pricing file.
        </footer>
      </body>
    </html>
  );
}
