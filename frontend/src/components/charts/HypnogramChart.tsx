"use client";

import { STAGE_COLORS } from "@/lib/theme";
import type { Prediction } from "@/types/analysis";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const STAGE_LABELS = ["W", "N1", "N2", "N3", "REM"];

interface HypnogramChartProps {
  predictions: Prediction[];
  title?: string;
  expertStages?: (string | null)[];
}

export function HypnogramChart({
  predictions,
  title = "Hypnogramme IA",
  expertStages,
}: HypnogramChartProps) {
  const data = predictions.map((p, i) => ({
    time: p.timeHours,
    stage: p.stageIndex,
    stageName: p.stage,
    expert: expertStages?.[i]
      ? STAGE_LABELS.indexOf(expertStages[i]!)
      : null,
  }));

  const remRegions = data.filter((d) => d.stage === 4);

  return (
    <div className="w-full">
      <h3 className="mb-3 text-sm font-semibold text-[var(--dps-primary)]">
        {title}
      </h3>
      <div className="dps-chart-wrap h-[280px] w-full rounded-lg border border-[var(--dps-border)] bg-[var(--dps-surface-2)] p-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="var(--dps-border)"
              opacity={0.5}
            />
            <XAxis
              dataKey="time"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(v) => `${v.toFixed(1)}h`}
              tick={{ fill: "var(--dps-text-muted)", fontSize: 11 }}
              axisLine={{ stroke: "var(--dps-border)" }}
            />
            <YAxis
              domain={[-0.5, 4.5]}
              ticks={[0, 1, 2, 3, 4]}
              tickFormatter={(v) => STAGE_LABELS[v] ?? ""}
              reversed
              tick={{ fill: "var(--dps-text-muted)", fontSize: 11 }}
              axisLine={{ stroke: "var(--dps-border)" }}
              width={42}
            />
            <Tooltip
              contentStyle={{
                background: "var(--dps-surface)",
                border: "1px solid var(--dps-border)",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={(value, name) => {
                if (name === "stage") return [STAGE_LABELS[Number(value)], "Stade IA"];
                return [value, name];
              }}
              labelFormatter={(l) => `Temps: ${Number(l).toFixed(2)} h`}
            />
            {remRegions.map((d, i) => (
              <ReferenceArea
                key={`rem-${i}`}
                x1={d.time}
                x2={d.time + 30 / 3600}
                fill={STAGE_COLORS.REM}
                fillOpacity={0.12}
              />
            ))}
            <Line
              type="stepAfter"
              dataKey="stage"
              stroke="var(--dps-primary)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <StageLegend />
    </div>
  );
}

export function ExpertHypnogramChart({
  expertStages,
  predictions,
}: {
  expertStages: (string | null)[];
  predictions: Prediction[];
}) {
  const data = predictions.map((p, i) => ({
    time: p.timeHours,
    stage:
      expertStages[i] != null
        ? STAGE_LABELS.indexOf(expertStages[i]!)
        : null,
  })).filter((d) => d.stage !== null && d.stage >= 0);

  if (data.length === 0) return null;

  return (
    <div className="mt-6 w-full">
      <h3 className="mb-3 text-sm font-semibold text-[var(--dps-primary)]">
        Hypnogramme expert (référence)
      </h3>
      <div className="h-[220px] w-full rounded-lg border border-[var(--dps-border)] bg-[var(--dps-surface-2)] p-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--dps-border)" opacity={0.5} />
            <XAxis
              dataKey="time"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(v) => `${v.toFixed(1)}h`}
              tick={{ fill: "var(--dps-text-muted)", fontSize: 11 }}
            />
            <YAxis
              domain={[-0.5, 4.5]}
              ticks={[0, 1, 2, 3, 4]}
              tickFormatter={(v) => STAGE_LABELS[v] ?? ""}
              reversed
              tick={{ fill: "var(--dps-text-muted)", fontSize: 11 }}
              width={42}
            />
            <Line
              type="stepAfter"
              dataKey="stage"
              stroke="var(--dps-accent)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function StageLegend() {
  return (
    <div className="mt-4 flex flex-wrap gap-4">
      {STAGE_LABELS.map((name) => (
        <div key={name} className="flex items-center gap-2 text-xs text-[var(--dps-text)]">
          <span
            className="inline-block h-3.5 w-3.5 rounded"
            style={{ background: STAGE_COLORS[name] }}
          />
          {name}
        </div>
      ))}
    </div>
  );
}
