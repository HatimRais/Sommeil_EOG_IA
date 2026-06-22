"use client";

import { Sidebar } from "@/components/layout/Sidebar";
import type { AnalysisResult } from "@/types/analysis";
import { Menu } from "lucide-react";
import { useEffect } from "react";

interface MobileChromeProps {
  open: boolean;
  onOpen: () => void;
  onClose: () => void;
  apiOnline: boolean | null;
  children: React.ReactNode;
  sidebarProps: React.ComponentProps<typeof Sidebar>;
}

/** UI téléphone uniquement — ne modifie pas le DOM desktop. */
export function MobileChrome({
  open,
  onOpen,
  onClose,
  apiOnline,
  children,
  sidebarProps,
}: MobileChromeProps) {
  useEffect(() => {
    document.documentElement.dataset.dpsPhone = "true";
    return () => {
      delete document.documentElement.dataset.dpsPhone;
    };
  }, []);

  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  return (
    <>
      {open && (
        <button
          type="button"
          className="dps-phone-backdrop"
          aria-label="Fermer le menu"
          onClick={onClose}
        />
      )}

      <div className="dps-phone-topbar">
        <button
          type="button"
          className="dps-phone-menu-btn"
          aria-label="Ouvrir le menu"
          aria-expanded={open}
          onClick={onOpen}
        >
          <Menu className="h-5 w-5" />
        </button>
        <span className="dps-phone-topbar-title">DeepSleep AI</span>
        <span
          className={`dps-phone-topbar-status ${
            apiOnline ? "text-[var(--dps-success)]" : "text-[var(--dps-danger)]"
          }`}
        >
          {apiOnline === null ? "…" : apiOnline ? "En ligne" : "Hors ligne"}
        </span>
      </div>

      <div className="dps-phone-shell flex h-dvh flex-col overflow-hidden">
        <Sidebar {...sidebarProps} phoneDrawer open={open} onPhoneClose={onClose} />
        {children}
      </div>
    </>
  );
}

export type { AnalysisResult };
