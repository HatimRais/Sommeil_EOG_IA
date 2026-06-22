"use client";

import { Card, StatusChip } from "@/components/ui/Card";
import type { PatientInfo } from "@/types/analysis";
import { Calendar, Clock, Cpu, FileAudio, Hash, Radio, Zap } from "lucide-react";

export function PatientCard({ patient }: { patient: PatientInfo }) {
  const fields = [
    { icon: Hash, label: "ID enregistrement", value: patient.id },
    { icon: Calendar, label: "Date", value: patient.recordingDate },
    { icon: Clock, label: "Durée", value: patient.durationFormatted },
    { icon: FileAudio, label: "Fichier source", value: patient.sourceFile },
    { icon: Radio, label: "Canal EOG", value: patient.eogChannel },
    {
      icon: Zap,
      label: "Échantillonnage",
      value: `${patient.sfreqOrig} → ${patient.sfreqTarget} Hz`,
    },
    { icon: Cpu, label: "Époques analysées", value: String(patient.nEpochs) },
    { icon: Zap, label: "Inférence", value: `${patient.inferenceMs} ms` },
  ];

  return (
    <Card accent className="animate-fade-in-up">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-8 dps-patient-grid">
        {fields.map(({ icon: Icon, label, value }) => (
          <div key={label} className="min-w-0">
            <div className="mb-1 flex items-center gap-1.5 text-[0.68rem] font-semibold uppercase tracking-wide text-[var(--dps-text-muted)]">
              <Icon className="h-3 w-3 shrink-0" />
              {label}
            </div>
            <div className="truncate text-sm font-medium text-[var(--dps-text)]" title={value}>
              {value}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

export function MetricsGrid({
  metrics,
  cycles,
}: {
  metrics: import("@/types/analysis").MetricCard[];
  cycles: number;
}) {
  return (
    <div className="animate-fade-in-up" style={{ animationDelay: "0.1s" }}>
      <h2 className="mb-4 border-b-2 border-[var(--dps-primary-mid)] pb-2 text-sm font-bold uppercase tracking-wide text-[var(--dps-primary)]">
        Métriques cliniques du sommeil
      </h2>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5 dps-metrics-grid">
        {metrics.map((m) => (
          <Card key={m.label} className="!p-4 transition-shadow hover:shadow-[var(--dps-shadow-lg)]">
            <div className="text-[0.72rem] font-medium text-[var(--dps-text-muted)]">
              {m.label}
            </div>
            <div className="mt-1 text-xl font-semibold text-[var(--dps-primary)]">
              {m.value}
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <StatusChip status={m.status} label={m.statusLabel} />
              <span className="text-[0.65rem] text-[var(--dps-text-muted)]">
                ref: {m.reference}
              </span>
            </div>
          </Card>
        ))}
        <Card className="!p-4">
          <div className="text-[0.72rem] font-medium text-[var(--dps-text-muted)]">
            Cycles de sommeil
          </div>
          <div className="mt-1 text-xl font-semibold text-[var(--dps-primary)]">
            {cycles}
          </div>
          <div className="mt-2">
            <StatusChip status="info" label="typique: 4–6" />
          </div>
        </Card>
      </div>
    </div>
  );
}
