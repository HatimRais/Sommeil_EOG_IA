"use client";

import { NORMS } from "@/lib/theme";
import { motion } from "framer-motion";
import {
  Activity,
  Brain,
  Moon,
  Shield,
  Sparkles,
  Stethoscope,
} from "lucide-react";

const FEATURES = [
  {
    icon: Brain,
    title: "IA CNN 1D",
    desc: "~88 % de précision vs expert humain",
  },
  {
    icon: Activity,
    title: "5 stades AASM",
    desc: "W · N1 · N2 · N3 · REM en époques de 30 s",
  },
  {
    icon: Sparkles,
    title: "OpenVINO NPU/GPU",
    desc: "Inférence accélérée jusqu'à 5 650 époques/s",
  },
  {
    icon: Shield,
    title: "Aide à la décision",
    desc: "Métriques cliniques et normes adultes intégrées",
  },
];

export function WelcomeLanding() {
  return (
    <div className="space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="dps-welcome-hero relative overflow-hidden rounded-2xl bg-gradient-to-br from-[var(--dps-header-from)] to-[var(--dps-header-to)] p-8 text-white shadow-[var(--dps-shadow-lg)] md:p-12"
      >
        <div className="absolute -right-8 -top-8 h-48 w-48 rounded-full bg-white/5" />
        <div className="absolute -bottom-12 -left-12 h-64 w-64 rounded-full bg-white/5" />
        <div className="relative">
          <div className="mb-4 flex items-center gap-3">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-white/15 backdrop-blur">
              <Moon className="h-7 w-7" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight md:text-3xl">
                Bienvenue sur DeepSleep AI
              </h1>
              <p className="mt-1 text-sm text-white/80">
                Laboratoire de polysomnographie · Scoring automatisé EOG
              </p>
            </div>
          </div>
          <p className="max-w-2xl text-sm leading-relaxed text-white/90 md:text-base">
            Cette plateforme effectue le scoring polysomnographique automatisé à partir d&apos;un
            seul canal <strong>EOG</strong>, selon la norme <strong>AASM 5 classes</strong>.
            Importez un enregistrement <code className="rounded bg-white/15 px-1.5 py-0.5 text-xs">.edf</code>,
            lancez l&apos;analyse et obtenez hypnogramme, architecture du sommeil et rapport clinique.
          </p>
        </div>
      </motion.div>

      <div className="dps-feature-grid grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {FEATURES.map(({ icon: Icon, title, desc }, i) => (
          <motion.div
            key={title}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 + i * 0.08 }}
            className="rounded-xl border border-[var(--dps-border)] bg-[var(--dps-surface)] p-5 shadow-[var(--dps-shadow)] transition-all hover:-translate-y-0.5 hover:shadow-[var(--dps-shadow-lg)]"
          >
            <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--dps-primary-soft)] text-[var(--dps-primary)]">
              <Icon className="h-5 w-5" />
            </div>
            <h3 className="font-semibold text-[var(--dps-text)]">{title}</h3>
            <p className="mt-1 text-sm text-[var(--dps-text-muted)]">{desc}</p>
          </motion.div>
        ))}
      </div>

      <div className="dps-workflow-grid grid gap-6 lg:grid-cols-3">
        <div className="rounded-xl border border-[var(--dps-border)] bg-[var(--dps-surface)] p-6 shadow-[var(--dps-shadow)] lg:col-span-2">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold text-[var(--dps-primary)]">
            <Stethoscope className="h-5 w-5" />
            Workflow d&apos;analyse
          </h2>
          <ol className="space-y-4">
            {[
              "Sélectionnez le matériel d'accélération (NPU, GPU ou CPU)",
              "Importez un fichier PSG .edf ou choisissez un patient de la base interne",
              "Optionnel : ajoutez l'hypnogramme expert pour la validation IA vs expert",
              "Cliquez sur « Analyser l'enregistrement » pour lancer le scoring",
            ].map((step, i) => (
              <li key={i} className="flex gap-4">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[var(--dps-primary)] text-sm font-bold text-white">
                  {i + 1}
                </span>
                <span className="pt-1 text-sm text-[var(--dps-text)]">{step}</span>
              </li>
            ))}
          </ol>
        </div>

        <div className="rounded-xl border border-[var(--dps-border)] bg-[var(--dps-surface)] p-6 shadow-[var(--dps-shadow)]">
          <h2 className="mb-4 text-sm font-bold uppercase tracking-wide text-[var(--dps-primary)]">
            Valeurs de référence (adulte)
          </h2>
          <ul className="space-y-3">
            {NORMS.map(({ label, range }) => (
              <li key={label}>
                <div className="text-sm font-medium text-[var(--dps-text)]">{label}</div>
                <div className="text-xs text-[var(--dps-text-muted)]">{range}</div>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="rounded-lg border-l-4 border-[var(--dps-warning)] bg-[var(--dps-chip-warn-bg)] p-4 text-sm text-[var(--dps-chip-warn-fg)]">
        <strong>Avertissement clinique.</strong> Cet outil est une{" "}
        <strong>aide à la décision</strong> pour les technologues du sommeil et les médecins.
        Il ne remplace pas le scoring visuel expert de la polysomnographie. Toutes les prédictions
        IA doivent être vérifiées par du personnel qualifié.
      </div>
    </div>
  );
}
