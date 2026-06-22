import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** CSV compatible Excel (locale FR : séparateur ; + BOM UTF-8). */
export function buildExcelCsv(rows: (string | number)[][]): string {
  const SEP = ";";
  const escape = (value: string | number) => {
    const s = String(value);
    if (s.includes(SEP) || s.includes('"') || s.includes("\n") || s.includes("\r")) {
      return `"${s.replace(/"/g, '""')}"`;
    }
    return s;
  };
  const body = rows.map((row) => row.map(escape).join(SEP)).join("\r\n");
  return `\uFEFF${body}`;
}

export function downloadExcelCsv(filename: string, rows: (string | number)[][]) {
  const blob = new Blob([buildExcelCsv(rows)], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
