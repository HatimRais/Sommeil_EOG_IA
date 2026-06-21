"use client";

import { STAGE_COLORS } from "@/lib/theme";
import type { ClinicalReport } from "@/types/analysis";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

export function ArchitecturePie({ clinical }: { clinical: ClinicalReport }) {
  const data = [
    { name: "N1", value: clinical.Counts.N1, color: STAGE_COLORS.N1 },
    { name: "N2", value: clinical.Counts.N2, color: STAGE_COLORS.N2 },
    { name: "N3", value: clinical.Counts.N3, color: STAGE_COLORS.N3 },
    { name: "REM", value: clinical.Counts.REM, color: STAGE_COLORS.REM },
  ].filter((d) => d.value > 0);

  const total = data.reduce((s, d) => s + d.value, 0);

  if (total === 0) {
    return (
      <div className="flex h-[300px] items-center justify-center text-sm text-[var(--dps-text-muted)]">
        Aucun sommeil détecté
      </div>
    );
  }

  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold text-[var(--dps-primary)]">
        Architecture du sommeil (% TST)
      </h3>
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={55}
              outerRadius={95}
              paddingAngle={2}
              dataKey="value"
              label={({ name, percent }) =>
                `${name} ${((percent ?? 0) * 100).toFixed(1)}%`
              }
              labelLine={{ stroke: "var(--dps-text-muted)" }}
            >
              {data.map((entry) => (
                <Cell key={entry.name} fill={entry.color} stroke="var(--dps-surface)" strokeWidth={2} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                background: "var(--dps-surface)",
                border: "1px solid var(--dps-border)",
                borderRadius: 8,
              }}
              formatter={(value, name) => {
                const n = Number(value ?? 0);
                return [
                  `${n} époques (${((n / total) * 100).toFixed(1)}%)`,
                  String(name),
                ];
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
