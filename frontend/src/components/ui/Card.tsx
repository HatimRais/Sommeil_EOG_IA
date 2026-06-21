import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

export function Card({
  children,
  className,
  accent = false,
}: {
  children: ReactNode;
  className?: string;
  accent?: boolean;
}) {
  return (
    <div
      className={cn(
        "rounded-xl border border-[var(--dps-border)] bg-[var(--dps-surface)] p-5 shadow-[var(--dps-shadow)]",
        accent && "border-l-4 border-l-[var(--dps-primary)]",
        className
      )}
    >
      {children}
    </div>
  );
}

export function StatusChip({
  status,
  label,
}: {
  status: "norm" | "warn" | "alert" | "info";
  label: string;
}) {
  const styles = {
    norm: "bg-[var(--dps-chip-norm-bg)] text-[var(--dps-chip-norm-fg)] border-[var(--dps-chip-norm-bd)]",
    warn: "bg-[var(--dps-chip-warn-bg)] text-[var(--dps-chip-warn-fg)] border-[var(--dps-chip-warn-bd)]",
    alert: "bg-[var(--dps-chip-alert-bg)] text-[var(--dps-chip-alert-fg)] border-[var(--dps-chip-alert-bd)]",
    info: "bg-[var(--dps-chip-info-bg)] text-[var(--dps-chip-info-fg)] border-[var(--dps-chip-info-bd)]",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-[0.68rem] font-bold tracking-wide",
        styles[status]
      )}
    >
      {label}
    </span>
  );
}
