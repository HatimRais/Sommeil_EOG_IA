"use client";

import type { AnalysisResult } from "@/types/analysis";
import { Cpu } from "lucide-react";

export function HeaderBar({ result }: { result?: AnalysisResult | null }) {
  const today = new Date().toISOString().slice(0, 10);
  const engine = result?.engine.runtimeDevice ?? "—";

  return (
    <header className="mb-6 flex flex-col gap-3 rounded-xl bg-gradient-to-r from-[var(--dps-header-from)] to-[var(--dps-header-to)] px-5 py-4 text-[var(--dps-header-text)] shadow-[var(--dps-shadow)] sm:flex-row sm:items-center sm:justify-between">
      <div>
        <h1 className="flex items-center gap-2 text-lg font-semibold md:text-xl">
          <span aria-hidden>⚕️</span>
          DeepSleep AI · Laboratoire de polysomnographie
        </h1>
        <p className="mt-0.5 text-xs text-white/75 md:text-sm">
          Scoring automatisé EOG · Norme AASM 5 classes
        </p>
      </div>
      <div className="flex flex-wrap gap-2">
        <span className="rounded-full border border-white/20 bg-white/10 px-3 py-1 text-xs">
          {today}
        </span>
        <span className="flex items-center gap-1.5 rounded-full border border-white/20 bg-white/10 px-3 py-1 text-xs">
          <Cpu className="h-3 w-3" />
          {engine}
        </span>
      </div>
    </header>
  );
}
