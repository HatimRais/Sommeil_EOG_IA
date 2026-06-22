"use client";

import { analyzeRecording, fetchDevices, fetchPatients } from "@/lib/api";
import type { AnalysisResult, DeviceOption, PatientRecord } from "@/types/analysis";
import { motion } from "framer-motion";
import {
  ChevronRight,
  Cpu,
  Database,
  Loader2,
  Moon,
  Monitor,
  Play,
  Sun,
  Upload,
  X,
} from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

interface SidebarProps {
  onAnalyze: (result: AnalysisResult) => void;
  onError: (msg: string) => void;
  onLoading: (loading: boolean) => void;
  loading: boolean;
  mobileOpen?: boolean;
  onMobileClose?: () => void;
}

export function Sidebar({
  onAnalyze,
  onError,
  onLoading,
  loading,
  mobileOpen = false,
  onMobileClose,
}: SidebarProps) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [devices, setDevices] = useState<DeviceOption[]>([]);
  const [patients, setPatients] = useState<PatientRecord[]>([]);
  const [device, setDevice] = useState("AUTO");
  const [source, setSource] = useState<"upload" | "database">("upload");
  const [signalFile, setSignalFile] = useState<File | null>(null);
  const [labelsFile, setLabelsFile] = useState<File | null>(null);
  const [patientId, setPatientId] = useState("");

  useEffect(() => {
    setMounted(true);
    fetchDevices()
      .then(setDevices)
      .catch(() => setDevices([{ id: "AUTO", label: "Auto-select" }]));
    fetchPatients()
      .then(setPatients)
      .catch(() => setPatients([]));
  }, []);

  const handleAnalyze = async () => {
    if (source === "upload" && !signalFile) {
      onError("Veuillez sélectionner un fichier signal .edf");
      return;
    }
    if (source === "database" && !patientId) {
      onError("Veuillez sélectionner un patient");
      return;
    }

    onLoading(true);
    onError("");
    try {
      const result = await analyzeRecording({
        device,
        signal: signalFile ?? undefined,
        labels: labelsFile ?? undefined,
        patientId: source === "database" ? patientId : undefined,
      });
      onAnalyze(result);
    } catch (e) {
      onError(e instanceof Error ? e.message : "Erreur d'analyse");
    } finally {
      onLoading(false);
    }
  };

  const themes = [
    { id: "light", icon: Sun, label: "Clair" },
    { id: "dark", icon: Moon, label: "Sombre" },
    { id: "system", icon: Monitor, label: "Système" },
  ] as const;

  return (
    <aside
      className={`dps-sidebar flex w-full shrink-0 flex-col border-r border-[var(--dps-sb-border)] bg-[var(--dps-sb-bg)] lg:h-full lg:min-h-0 lg:w-80${mobileOpen ? " dps-sidebar-open" : ""}`}
    >
      <div className="border-b border-[var(--dps-sb-border)] p-5">
        <div className="flex items-center gap-3">
          <span className="text-3xl" aria-hidden>
            ⚕️
          </span>
          <div className="min-w-0 flex-1">
            <h2 className="text-lg font-bold tracking-tight text-[var(--dps-sb-text)]">
              DeepSleep AI
            </h2>
            <p className="text-xs text-[var(--dps-sb-muted)]">
              Polysomnography Lab · v2.0
            </p>
          </div>
          <button
            type="button"
            className="dps-sidebar-close"
            aria-label="Fermer le menu"
            onClick={onMobileClose}
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {["EOG", "AASM 5", "OpenVINO"].map((b) => (
            <span
              key={b}
              className="rounded-full border border-[var(--dps-sb-border)] bg-[var(--dps-primary-soft)] px-2 py-0.5 text-[0.64rem] font-semibold text-[var(--dps-sb-accent)]"
            >
              {b}
            </span>
          ))}
        </div>
      </div>

      <div className="dps-sidebar-scroll flex-1 space-y-5 overflow-y-auto p-5">
        {/* Theme */}
        <section>
          <h3 className="mb-2 text-[0.72rem] font-bold uppercase tracking-widest text-[var(--dps-sb-accent)]">
            Apparence
          </h3>
          {mounted && (
            <div className="flex rounded-lg border border-[var(--dps-sb-border)] bg-[var(--dps-sb-surface)] p-1">
              {themes.map(({ id, icon: Icon, label }) => (
                <button
                  key={id}
                  onClick={() => setTheme(id)}
                  title={label}
                  className={`flex flex-1 items-center justify-center rounded-md py-2 transition-colors ${
                    theme === id
                      ? "bg-[var(--dps-sb-accent)] text-[var(--dps-sb-bg)]"
                      : "text-[var(--dps-sb-text)] hover:bg-[var(--dps-primary-soft)]"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                </button>
              ))}
            </div>
          )}
        </section>

        {/* Device */}
        <section>
          <h3 className="mb-2 text-[0.72rem] font-bold uppercase tracking-widest text-[var(--dps-sb-accent)]">
            Accélération matérielle
          </h3>
          <div className="relative">
            <Cpu className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--dps-sb-muted)]" />
            <select
              value={device}
              onChange={(e) => setDevice(e.target.value)}
              className="w-full appearance-none rounded-lg border border-[var(--dps-sb-border)] bg-[var(--dps-sb-surface)] py-2.5 pl-10 pr-8 text-sm text-[var(--dps-sb-text)] focus:outline-none focus:ring-2 focus:ring-[var(--dps-sb-accent)]"
            >
              {devices.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
          </div>
        </section>

        {/* Data source */}
        <section>
          <h3 className="mb-2 text-[0.72rem] font-bold uppercase tracking-widest text-[var(--dps-sb-accent)]">
            Données patient
          </h3>
          <div className="mb-3 flex rounded-lg border border-[var(--dps-sb-border)] bg-[var(--dps-sb-surface)] p-1">
            {(
              [
                { id: "upload" as const, icon: Upload, label: "Import" },
                { id: "database" as const, icon: Database, label: "Base" },
              ] as const
            ).map(({ id, icon: Icon, label }) => (
              <button
                key={id}
                onClick={() => setSource(id)}
                className={`flex flex-1 items-center justify-center gap-1.5 rounded-md py-2 text-xs font-medium transition-colors ${
                  source === id
                    ? "bg-[var(--dps-primary-soft)] text-[var(--dps-sb-accent)] ring-1 ring-[var(--dps-sb-accent)]"
                    : "text-[var(--dps-sb-text)]"
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </button>
            ))}
          </div>

          {source === "upload" ? (
            <div className="space-y-3">
              <label className="block">
                <span className="mb-1 block text-xs text-[var(--dps-sb-muted)]">
                  Signal PSG (.edf)
                </span>
                <input
                  type="file"
                  accept=".edf"
                  onChange={(e) => setSignalFile(e.target.files?.[0] ?? null)}
                  className="w-full rounded-lg border border-dashed border-[var(--dps-sb-border)] bg-[var(--dps-sb-surface)] px-3 py-2 text-xs text-[var(--dps-sb-text)] file:mr-2 file:rounded file:border-0 file:bg-[var(--dps-sb-accent)] file:px-2 file:py-1 file:text-xs file:text-white"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-xs text-[var(--dps-sb-muted)]">
                  Hypnogramme expert (.edf) — optionnel
                </span>
                <input
                  type="file"
                  accept=".edf"
                  onChange={(e) => setLabelsFile(e.target.files?.[0] ?? null)}
                  className="w-full rounded-lg border border-dashed border-[var(--dps-sb-border)] bg-[var(--dps-sb-surface)] px-3 py-2 text-xs text-[var(--dps-sb-text)] file:mr-2 file:rounded file:border-0 file:bg-[var(--dps-sb-accent)] file:px-2 file:py-1 file:text-xs file:text-white"
                />
              </label>
            </div>
          ) : (
            <select
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              className="w-full rounded-lg border border-[var(--dps-sb-border)] bg-[var(--dps-sb-surface)] px-3 py-2.5 text-sm text-[var(--dps-sb-text)]"
            >
              <option value="">— Sélectionner un patient —</option>
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.id} {p.hasLabels ? "✓ labels" : ""}
                </option>
              ))}
            </select>
          )}
        </section>
      </div>

      <div className="border-t border-[var(--dps-sb-border)] p-5">
        <motion.button
          whileTap={{ scale: 0.98 }}
          onClick={handleAnalyze}
          disabled={loading}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-[var(--dps-primary)] py-3.5 text-sm font-bold text-white shadow-[var(--dps-shadow)] transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Analyse en cours…
            </>
          ) : (
            <>
              <Play className="h-4 w-4" />
              Analyser l&apos;enregistrement
              <ChevronRight className="h-4 w-4" />
            </>
          )}
        </motion.button>
        <div className="mt-4 space-y-1 text-[0.68rem] text-[var(--dps-sb-muted)]">
          <p>Modèle : 1D-CNN (455k params, FP16)</p>
          <p>Norme : AASM 5 classes · 30 s @ 100 Hz</p>
        </div>
      </div>
    </aside>
  );
}
