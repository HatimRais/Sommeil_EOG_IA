"use client";

import { HeaderBar } from "@/components/layout/Header";
import { Sidebar } from "@/components/layout/Sidebar";
import { WelcomeLanding } from "@/components/landing/Welcome";
import { CnnArchitectureLoader } from "@/components/loading/CnnArchitectureLoader";
import { MetricsGrid, PatientCard } from "@/components/results/PatientCard";
import { ResultsTabs } from "@/components/results/ResultsTabs";
import { checkHealth } from "@/lib/api";
import type { AnalysisResult } from "@/types/analysis";
import { AlertCircle, Menu, Wifi, WifiOff } from "lucide-react";
import { useEffect, useState } from "react";

export default function DashboardPage() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  useEffect(() => {
    checkHealth().then(setApiOnline);
    const interval = setInterval(() => checkHealth().then(setApiOnline), 15000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    document.body.style.overflow = mobileNavOpen ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [mobileNavOpen]);

  const handleAnalyze = (analysis: AnalysisResult) => {
    setResult(analysis);
    setMobileNavOpen(false);
  };

  return (
    <div className="dps-app flex h-dvh flex-col overflow-hidden lg:flex-row">
      {mobileNavOpen && (
        <button
          type="button"
          className="dps-mobile-backdrop"
          aria-label="Fermer le menu"
          onClick={() => setMobileNavOpen(false)}
        />
      )}

      <div className="dps-mobile-topbar">
        <button
          type="button"
          className="dps-mobile-menu-btn"
          aria-label="Ouvrir le menu"
          aria-expanded={mobileNavOpen}
          onClick={() => setMobileNavOpen(true)}
        >
          <Menu className="h-5 w-5" />
        </button>
        <span className="dps-mobile-topbar-title">DeepSleep AI</span>
        <span
          className={`dps-mobile-topbar-status ${
            apiOnline ? "text-[var(--dps-success)]" : "text-[var(--dps-danger)]"
          }`}
        >
          {apiOnline === null ? "…" : apiOnline ? "En ligne" : "Hors ligne"}
        </span>
      </div>

      <Sidebar
        mobileOpen={mobileNavOpen}
        onMobileClose={() => setMobileNavOpen(false)}
        onAnalyze={handleAnalyze}
        onError={setError}
        onLoading={setLoading}
        loading={loading}
      />

      <main className="dps-main min-h-0 flex-1 overflow-y-auto">
        <div className="dps-main-inner mx-auto max-w-6xl p-4 md:p-6 lg:p-8">
          <HeaderBar result={result} />

          {apiOnline === false && (
            <div className="dps-alert-banner mb-4 flex items-center gap-3 rounded-lg border border-[var(--dps-chip-alert-bd)] bg-[var(--dps-chip-alert-bg)] p-4 text-sm text-[var(--dps-chip-alert-fg)]">
              <WifiOff className="h-5 w-5 shrink-0" />
              <div>
                <strong>API hors ligne.</strong> En local, démarrez le backend :{" "}
                <code className="rounded bg-black/10 px-1.5 py-0.5 text-xs">
                  uvicorn src.api.main:app --reload --port 8000
                </code>
                {" "}puis relancez{" "}
                <code className="rounded bg-black/10 px-1.5 py-0.5 text-xs">
                  npm run dev
                </code>
                {" "}dans <code className="text-xs">frontend/</code>.
              </div>
            </div>
          )}

          {apiOnline === true && (
            <div className="dps-api-status mb-4 flex items-center gap-2 text-xs text-[var(--dps-success)]">
              <Wifi className="h-3.5 w-3.5" />
              API connectée
            </div>
          )}

          {error && (
            <div className="dps-alert-banner mb-4 flex items-center gap-3 rounded-lg border border-[var(--dps-chip-alert-bd)] bg-[var(--dps-chip-alert-bg)] p-4 text-sm text-[var(--dps-chip-alert-fg)]">
              <AlertCircle className="h-5 w-5 shrink-0" />
              {error}
            </div>
          )}

          {loading && <CnnArchitectureLoader />}

          {!loading && !result && <WelcomeLanding />}

          {!loading && result && (
            <div className="dps-results-section space-y-8">
              <PatientCard patient={result.patient} />
              <MetricsGrid metrics={result.metrics} cycles={result.cycles} />
              <ResultsTabs result={result} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
