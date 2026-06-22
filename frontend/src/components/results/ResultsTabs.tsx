"use client";

import { ArchitecturePie } from "@/components/charts/ArchitecturePie";
import { ConfusionMatrix } from "@/components/charts/ConfusionMatrix";
import {
  ExpertHypnogramChart,
  HypnogramChart,
} from "@/components/charts/HypnogramChart";
import { Card, StatusChip } from "@/components/ui/Card";
import { STAGE_FULL } from "@/lib/theme";
import { downloadExcelCsv } from "@/lib/utils";
import type { AnalysisResult } from "@/types/analysis";
import { motion } from "framer-motion";
import {
  Activity,
  BarChart3,
  ClipboardList,
  Download,
  FileCheck,
  Settings,
} from "lucide-react";
import { useState } from "react";

const TABS = [
  { id: "hypno", label: "Hypnogramme", icon: Activity },
  { id: "arch", label: "Architecture", icon: BarChart3 },
  { id: "valid", label: "IA vs Expert", icon: FileCheck },
  { id: "report", label: "Rapport clinique", icon: ClipboardList },
  { id: "tech", label: "Technique", icon: Settings },
] as const;

type TabId = (typeof TABS)[number]["id"];

export function ResultsTabs({ result }: { result: AnalysisResult }) {
  const [active, setActive] = useState<TabId>("hypno");

  const downloadCsv = () => {
    const rows: (string | number)[][] = [
      ["epoch", "time_hms", "stage_AI", "stage_expert", "agreement"],
      ...result.predictions.map((p, i) => {
        const expert = result.validation?.expertStages?.[i] ?? "";
        const agree =
          expert && expert === p.stage ? "OK" : expert ? "MISMATCH" : "";
        const h = Math.floor((p.epoch * 30) / 3600);
        const m = Math.floor(((p.epoch * 30) % 3600) / 60);
        const s = Math.floor((p.epoch * 30) % 60);
        const time = `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
        return [p.epoch, time, p.stage, expert, agree];
      }),
    ];
    downloadExcelCsv(`${result.patient.id}_hypnogram.csv`, rows);
  };

  return (
    <div className="animate-fade-in-up" style={{ animationDelay: "0.2s" }}>
      <div className="mb-4 flex gap-1 overflow-x-auto border-b border-[var(--dps-border)] pb-px">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActive(id)}
            className={`flex shrink-0 items-center gap-2 rounded-t-lg px-4 py-2.5 text-sm font-medium transition-colors ${
              active === id
                ? "border-b-2 border-[var(--dps-primary)] text-[var(--dps-primary)]"
                : "text-[var(--dps-text-muted)] hover:text-[var(--dps-text)]"
            }`}
          >
            <Icon className="h-4 w-4" />
            {label}
          </button>
        ))}
      </div>

      <motion.div
        key={active}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
      >
        {active === "hypno" && (
          <Card>
            <HypnogramChart predictions={result.predictions} />
            {result.validation?.expertStages && (
              <ExpertHypnogramChart
                expertStages={result.validation.expertStages}
                predictions={result.predictions}
              />
            )}
          </Card>
        )}

        {active === "arch" && (
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <ArchitecturePie clinical={result.clinical} />
            </Card>
            <Card>
              <h3 className="mb-4 text-sm font-semibold text-[var(--dps-primary)]">
                Durées par stade
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-[var(--dps-border)] text-left text-[var(--dps-text-muted)]">
                      <th className="pb-2 pr-4">Stade</th>
                      <th className="pb-2 pr-4">Époques</th>
                      <th className="pb-2 pr-4">Durée</th>
                      <th className="pb-2">%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.stages.short.map((name, i) => {
                      const n = result.clinical.Counts[name];
                      const mins = (n * 30) / 60;
                      const h = Math.floor(mins / 60);
                      const m = Math.round(mins % 60);
                      const pct =
                        i === 0
                          ? (mins / result.clinical.TIB_min) * 100
                          : (mins / Math.max(1, result.clinical.TST_min)) * 100;
                      return (
                        <tr key={name} className="border-b border-[var(--dps-border-soft)]">
                          <td className="py-2 pr-4 font-medium">{STAGE_FULL[name]}</td>
                          <td className="py-2 pr-4">{n}</td>
                          <td className="py-2 pr-4">{h}h {String(m).padStart(2, "0")}min</td>
                          <td className="py-2">{pct.toFixed(1)} %</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="mt-6 space-y-2 text-sm text-[var(--dps-text-muted)]">
                <p>• Cycles détectés : <strong className="text-[var(--dps-text)]">{result.cycles}</strong></p>
                <p>• Transitions de stade : <strong className="text-[var(--dps-text)]">{result.transitions}</strong></p>
                <p>• Indice de fragmentation : <strong className="text-[var(--dps-text)]">{result.fragmentation.toFixed(1)}</strong></p>
                <p>• Confiance moyenne IA : <strong className="text-[var(--dps-text)]">{result.meanConfidence.toFixed(1)} %</strong></p>
              </div>
            </Card>
          </div>
        )}

        {active === "valid" && (
          <Card>
            {!result.validation ? (
              <p className="text-sm text-[var(--dps-text-muted)]">
                Importez un hypnogramme expert (.edf) pour activer les métriques de validation.
              </p>
            ) : result.validation.parseError ? (
              <p className="text-sm text-[var(--dps-warning)]">
                L&apos;hypnogramme expert n&apos;a pas pu être analysé.
              </p>
            ) : (
              <div className="grid gap-6 lg:grid-cols-2">
                <ConfusionMatrix validation={result.validation} />
                <div>
                  <h3 className="mb-3 text-sm font-semibold text-[var(--dps-primary)]">
                    Performance par stade
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-[var(--dps-border)] text-left text-[var(--dps-text-muted)]">
                          <th className="pb-2 pr-3">Stade</th>
                          <th className="pb-2 pr-3">Sensibilité</th>
                          <th className="pb-2 pr-3">Précision</th>
                          <th className="pb-2 pr-3">F1</th>
                          <th className="pb-2">Support</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.validation.perStage?.map((row) => (
                          <tr key={row.stage} className="border-b border-[var(--dps-border-soft)]">
                            <td className="py-2 pr-3 font-medium">{row.stage}</td>
                            <td className="py-2 pr-3">{row.sensitivity}%</td>
                            <td className="py-2 pr-3">{row.precision}%</td>
                            <td className="py-2 pr-3">{row.f1}%</td>
                            <td className="py-2">{row.support}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="mt-6 grid grid-cols-2 gap-4">
                    <div className="rounded-lg border border-[var(--dps-border)] bg-[var(--dps-surface-2)] p-4">
                      <div className="text-xs text-[var(--dps-text-muted)]">Précision globale</div>
                      <div className="text-2xl font-bold text-[var(--dps-primary)]">
                        {result.validation.accuracy?.toFixed(2)} %
                      </div>
                    </div>
                    <div className="rounded-lg border border-[var(--dps-border)] bg-[var(--dps-surface-2)] p-4">
                      <div className="text-xs text-[var(--dps-text-muted)]">Cohen&apos;s κ</div>
                      <div className="text-2xl font-bold text-[var(--dps-primary)]">
                        {result.validation.kappa?.toFixed(3)}
                      </div>
                      <div className="mt-1 text-xs text-[var(--dps-text-muted)]">
                        {result.validation.kappaLabel}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </Card>
        )}

        {active === "report" && (
          <div className="space-y-6">
            <Card>
              <h3 className="mb-4 text-sm font-semibold text-[var(--dps-primary)]">
                Interprétation automatique
              </h3>
              <div className="space-y-3">
                {result.interpretation.map((item, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-3 rounded-lg border-l-[3px] bg-[var(--dps-surface-2)] p-3"
                    style={{
                      borderLeftColor:
                        item.severity === "norm"
                          ? "var(--dps-success)"
                          : item.severity === "warn"
                            ? "var(--dps-warning)"
                            : "var(--dps-danger)",
                    }}
                  >
                    <StatusChip status={item.severity} label={item.severity.toUpperCase()} />
                    <p className="text-sm text-[var(--dps-text)]">{item.text}</p>
                  </div>
                ))}
              </div>
            </Card>
            <Card>
              <div className="mb-4 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-[var(--dps-primary)]">
                  Résumé quantitatif
                </h3>
                <button
                  onClick={downloadCsv}
                  className="flex items-center gap-2 rounded-lg bg-[var(--dps-primary)] px-4 py-2 text-sm font-semibold text-white transition-opacity hover:opacity-90"
                >
                  <Download className="h-4 w-4" />
                  Exporter CSV
                </button>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-[var(--dps-border)] text-left text-[var(--dps-text-muted)]">
                      <th className="pb-2 pr-4">Métrique</th>
                      <th className="pb-2 pr-4">Valeur</th>
                      <th className="pb-2">Référence</th>
                    </tr>
                  </thead>
                  <tbody className="text-[var(--dps-text)]">
                    {[
                      ["Temps de sommeil total", result.metrics[0]?.value, "6.5 – 8h"],
                      ["Efficacité du sommeil", result.metrics[1]?.value, "≥ 85 %"],
                      ["Latence d'endormissement", result.metrics[2]?.value, "< 20 min"],
                      ["Latence REM", result.metrics[4]?.value, "70 – 120 min"],
                      ["WASO", result.metrics[3]?.value, "< 30 min"],
                      ["REM % TST", result.metrics[5]?.value, "20 – 25 %"],
                      ["N3 % TST", result.metrics[6]?.value, "13 – 23 %"],
                      ["Éveils", result.metrics[8]?.value, "< 10"],
                      ["Cycles", String(result.cycles), "4 – 6"],
                    ].map(([metric, value, ref]) => (
                      <tr key={metric} className="border-b border-[var(--dps-border-soft)]">
                        <td className="py-2 pr-4">{metric}</td>
                        <td className="py-2 pr-4 font-medium">{value}</td>
                        <td className="py-2 text-[var(--dps-text-muted)]">{ref}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        )}

        {active === "tech" && (
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <h3 className="mb-4 text-sm font-semibold text-[var(--dps-primary)]">
                Acquisition du signal
              </h3>
              <dl className="space-y-2 text-sm">
                {Object.entries({
                  Fichier: result.patient.sourceFile,
                  "ID anonyme": result.patient.id,
                  Date: result.patient.recordingDate,
                  "Canal EOG": result.patient.eogChannel,
                  "Fréquence originale": `${result.patient.sfreqOrig} Hz`,
                  Resampling: `${result.patient.sfreqTarget} Hz`,
                  "Filtre passe-bande": result.technical.bandpass,
                  Normalisation: result.technical.normalization,
                  "Longueur d'époque": result.technical.epochLength,
                  "Total époques": result.patient.nEpochs,
                  Durée: result.patient.durationFormatted,
                }).map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-4 border-b border-[var(--dps-border-soft)] py-1.5">
                    <dt className="text-[var(--dps-text-muted)]">{k}</dt>
                    <dd className="font-medium text-[var(--dps-text)]">{v}</dd>
                  </div>
                ))}
              </dl>
            </Card>
            <Card>
              <h3 className="mb-4 text-sm font-semibold text-[var(--dps-primary)]">
                Moteur d&apos;inférence
              </h3>
              <dl className="space-y-2 text-sm">
                {Object.entries({
                  Modèle: result.technical.model,
                  Format: result.technical.format,
                  Paramètres: result.technical.parameters,
                  "Forme d'entrée": result.technical.inputShape,
                  Périphérique: result.engine.device,
                  "Runtime OpenVINO": result.engine.runtimeDevice,
                  Matériel: result.engine.hardware ?? "—",
                  "Temps d'inférence": `${result.engine.inferenceMs} ms`,
                  Débit: `${result.engine.throughput} époques/s`,
                  "Confiance moyenne": `${result.engine.meanConfidence} %`,
                }).map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-4 border-b border-[var(--dps-border-soft)] py-1.5">
                    <dt className="text-[var(--dps-text-muted)]">{k}</dt>
                    <dd className="font-medium text-[var(--dps-text)]">{v}</dd>
                  </div>
                ))}
              </dl>
            </Card>
            <div className="col-span-full rounded-lg border-l-4 border-[var(--dps-warning)] bg-[var(--dps-chip-warn-bg)] p-4 text-sm text-[var(--dps-chip-warn-fg)] lg:col-span-2">
              <strong>Limitations.</strong> Le modèle utilise un seul canal EOG et a été entraîné sur
              des enregistrements adultes. Les performances peuvent varier chez les enfants ou en
              présence de pathologies sévères. Les décisions cliniques doivent reposer sur le scoring
              visuel expert de la polysomnographie complète.
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
}
