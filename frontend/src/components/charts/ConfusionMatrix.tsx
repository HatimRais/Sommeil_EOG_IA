"use client";

import type { ValidationResult } from "@/types/analysis";

export function ConfusionMatrix({ validation }: { validation: ValidationResult }) {
  const cm = validation.confusion;
  if (!cm) return null;

  const maxVal = Math.max(...cm.matrixNorm.flat());

  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold text-[var(--dps-primary)]">
        Matrice de confusion (lignes en %)
      </h3>
      <div className="overflow-x-auto rounded-lg border border-[var(--dps-border)]">
        <table className="w-full min-w-[360px] border-collapse text-center text-xs">
          <thead>
            <tr>
              <th className="border border-[var(--dps-border)] bg-[var(--dps-surface-2)] p-2 text-[var(--dps-text-muted)]">
                Expert ↓ / IA →
              </th>
              {cm.stages.map((s) => (
                <th
                  key={s}
                  className="border border-[var(--dps-border)] bg-[var(--dps-surface-2)] p-2 font-semibold text-[var(--dps-text)]"
                >
                  {s}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cm.stages.map((rowStage, i) => (
              <tr key={rowStage}>
                <td className="border border-[var(--dps-border)] bg-[var(--dps-surface-2)] p-2 font-semibold">
                  {rowStage}
                </td>
                {cm.stages.map((_, j) => {
                  const pct = cm.matrixNorm[i][j];
                  const count = cm.matrix[i][j];
                  const intensity = maxVal > 0 ? pct / maxVal : 0;
                  return (
                    <td
                      key={j}
                      className="border border-[var(--dps-border)] p-2"
                      style={{
                        background: `color-mix(in srgb, var(--dps-primary) ${Math.round(intensity * 70)}%, var(--dps-surface))`,
                      }}
                    >
                      <div className="font-bold text-[var(--dps-text)]">
                        {pct.toFixed(0)}%
                      </div>
                      <div className="text-[0.65rem] text-[var(--dps-text-muted)]">
                        n={count}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
