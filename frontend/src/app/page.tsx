"use client";

import { HeaderBar } from "@/components/layout/Header";
import { Sidebar } from "@/components/layout/Sidebar";
import { WelcomeLanding } from "@/components/landing/Welcome";
import { CnnArchitectureLoader } from "@/components/loading/CnnArchitectureLoader";
import { MetricsGrid, PatientCard } from "@/components/results/PatientCard";
import { ResultsTabs } from "@/components/results/ResultsTabs";
import { checkHealth } from "@/lib/api";
import type { AnalysisResult } from "@/types/analysis";
import { AlertCircle, Wifi, WifiOff } from "lucide-react";
import { useEffect, useState } from "react";

export default function DashboardPage() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  useEffect(() => {
    checkHealth().then(setApiOnline);
    const interval = setInterval(() => checkHealth().then(setApiOnline), 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex min-h-screen flex-col lg:flex-row">
      <Sidebar
        onAnalyze={setResult}
        onError={setError}
        onLoading={setLoading}
        loading={loading}
      />

      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-6xl p-4 md:p-6 lg:p-8">
          <HeaderBar result={result} />

          {apiOnline === false && (
            <div className="mb-4 flex items-center gap-3 rounded-lg border border-[var(--dps-chip-alert-bd)] bg-[var(--dps-chip-alert-bg)] p-4 text-sm text-[var(--dps-chip-alert-fg)]">
              <WifiOff className="h-5 w-5 shrink-0" />
              <div>
                <strong>API hors ligne.</strong>{" "}
                {process.env.NEXT_PUBLIC_API_URL &&
                !/localhost|127\.0\.0\.1/.test(process.env.NEXT_PUBLIC_API_URL) ? (
                  <>
                    Impossible de joindre le backend à{" "}
                    <code className="rounded bg-black/10 px-1.5 py-0.5 text-xs">
                      {process.env.NEXT_PUBLIC_API_URL}
                    </code>
                    . Vérifiez que l&apos;API Python est déployée et accessible en HTTPS.
                  </>
                ) : (
                  <>
                    Démarrez le backend Python :{" "}
                    <code className="rounded bg-black/10 px-1.5 py-0.5 text-xs">
                      uvicorn src.api.main:app --reload --port 8000
                    </code>
                  </>
                )}
              </div>
            </div>
          )}

          {apiOnline === true && (
            <div className="mb-4 flex items-center gap-2 text-xs text-[var(--dps-success)]">
              <Wifi className="h-3.5 w-3.5" />
              API connectée
            </div>
          )}

          {error && (
            <div className="mb-4 flex items-center gap-3 rounded-lg border border-[var(--dps-chip-alert-bd)] bg-[var(--dps-chip-alert-bg)] p-4 text-sm text-[var(--dps-chip-alert-fg)]">
              <AlertCircle className="h-5 w-5 shrink-0" />
              {error}
            </div>
          )}

          {loading && <CnnArchitectureLoader />}

          {!loading && !result && <WelcomeLanding />}

          {!loading && result && (
            <div className="space-y-8">
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
