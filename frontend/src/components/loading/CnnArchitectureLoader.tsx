"use client";

import { STAGE_COLORS } from "@/lib/theme";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";

const PHASES = [
  "Lecture du signal EOG…",
  "Prétraitement · filtre 0,5–35 Hz · z-score",
  "Extraction des motifs convolutifs (4 blocs)",
  "Inférence OpenVINO · batch 64 époques",
  "Classification AASM · 5 stades",
  "Construction de l'hypnogramme…",
];

const CONV_BLOCKS = [
  { label: "Conv₁", filters: 64, kernel: 11, pool: 4, maps: 4, w: 88 },
  { label: "Conv₂", filters: 128, kernel: 7, pool: 4, maps: 5, w: 62 },
  { label: "Conv₃", filters: 256, kernel: 5, pool: 4, maps: 6, w: 44 },
  { label: "Conv₄", filters: 256, kernel: 3, pool: 2, maps: 6, w: 30 },
];

const STAGES = ["W", "N1", "N2", "N3", "REM"] as const;

function FeatureMapStack({
  x,
  y,
  w,
  h,
  maps,
  delay,
  accent,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  maps: number;
  delay: number;
  accent: string;
}) {
  return (
    <g>
      {Array.from({ length: maps }).map((_, i) => (
        <motion.rect
          key={i}
          x={x + i * 5}
          y={y - i * 6}
          width={w}
          height={h}
          rx={4}
          fill={`url(#fmGrad-${accent})`}
          stroke="var(--dps-primary)"
          strokeWidth={0.6}
          strokeOpacity={0.35}
          initial={{ opacity: 0, scale: 0.85 }}
          animate={{ opacity: 0.55 + i * 0.08, scale: 1 }}
          transition={{ delay: delay + i * 0.08, duration: 0.5 }}
          className="cnn-feature-map"
        />
      ))}
    </g>
  );
}

function CnnArchitectureSvg() {
  const vbW = 920;
  const vbH = 340;
  const baseY = 200;

  const stagePositions: { stage: (typeof STAGES)[number]; cx: number; cy: number }[] = [
    { stage: "W", cx: 748, cy: baseY - 30 },
    { stage: "N1", cx: 808, cy: baseY - 30 },
    { stage: "N2", cx: 868, cy: baseY - 30 },
    { stage: "N3", cx: 778, cy: baseY + 38 },
    { stage: "REM", cx: 838, cy: baseY + 38 },
  ];

  return (
    <svg
      viewBox={`0 0 ${vbW} ${vbH}`}
      className="cnn-arch-svg w-full max-w-4xl"
      role="img"
      aria-label="Architecture CNN 1D en cours d'inférence"
    >
      <defs>
        <linearGradient id="cnnBgGlow" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="var(--dps-primary)" stopOpacity="0.12" />
          <stop offset="50%" stopColor="var(--dps-accent)" stopOpacity="0.06" />
          <stop offset="100%" stopColor="var(--dps-primary)" stopOpacity="0.14" />
        </linearGradient>
        <linearGradient id="waveGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="var(--dps-accent)" stopOpacity="0.3" />
          <stop offset="50%" stopColor="var(--dps-primary)" />
          <stop offset="100%" stopColor="var(--dps-accent)" stopOpacity="0.3" />
        </linearGradient>
        <linearGradient id="fmGrad-a" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#5eb8a8" stopOpacity="0.55" />
          <stop offset="100%" stopColor="#0c4a6e" stopOpacity="0.75" />
        </linearGradient>
        <linearGradient id="fmGrad-b" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#93c5e8" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#1e5f8a" stopOpacity="0.8" />
        </linearGradient>
        <linearGradient id="fmGrad-c" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#3b82b6" stopOpacity="0.45" />
          <stop offset="100%" stopColor="#0c4a6e" stopOpacity="0.85" />
        </linearGradient>
        <linearGradient id="fmGrad-d" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#1e5f8a" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#0c4a6e" stopOpacity="0.9" />
        </linearGradient>
        <filter id="cnnGlow" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <marker id="arrowHead" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="var(--dps-primary)" fillOpacity="0.45" />
        </marker>
        <radialGradient id="nodePulse" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="var(--dps-accent)" stopOpacity="0.9" />
          <stop offset="100%" stopColor="var(--dps-primary)" stopOpacity="0.2" />
        </radialGradient>
      </defs>

      <ellipse cx={460} cy={168} rx={420} ry={130} fill="url(#cnnBgGlow)" className="cnn-ambient" />

      <path
        d="M 118 200 C 210 198, 280 202, 360 200 S 540 198, 640 200 S 740 202, 820 200"
        fill="none"
        stroke="var(--dps-primary)"
        strokeWidth={1.5}
        strokeOpacity={0.12}
        strokeDasharray="8 12"
        className="cnn-backbone"
      />

      {[0, 1, 2, 3, 4, 5].map((i) => (
        <circle key={i} r={i % 2 === 0 ? 3.5 : 2} fill="var(--dps-accent)" className="cnn-particle" opacity={0.85}>
          <animateMotion
            dur={`${2.4 + i * 0.25}s`}
            repeatCount="indefinite"
            begin={`${i * 0.4}s`}
            path="M 118 200 C 210 198, 280 202, 360 200 S 540 198, 640 200 S 740 202, 820 200"
          />
        </circle>
      ))}

      <g>
        <rect x={18} y={baseY - 76} width={100} height={152} rx={12} fill="var(--dps-surface)" stroke="var(--dps-border)" strokeWidth={1} opacity={0.95} />
        <text x={68} y={baseY - 86} textAnchor="middle" className="cnn-label">Entrée EOG</text>
        <text x={68} y={baseY - 70} textAnchor="middle" className="cnn-sublabel">3000 × 1 · 100 Hz</text>
        <path
          d="M 30 198 C 38 168, 46 198, 54 198 S 70 178, 78 208 S 90 188, 102 202"
          fill="none"
          stroke="url(#waveGrad)"
          strokeWidth={2.4}
          strokeLinecap="round"
          className="cnn-wave-path"
        />
        <path
          d="M 30 202 C 42 232, 54 202, 66 202 S 82 222, 94 192 S 106 212, 110 200"
          fill="none"
          stroke="url(#waveGrad)"
          strokeWidth={1.2}
          strokeOpacity={0.35}
          strokeLinecap="round"
          className="cnn-wave-path-alt"
        />
        <line x1={118} y1={baseY + 18} x2={138} y2={baseY + 18} stroke="var(--dps-primary)" strokeWidth={1.5} strokeOpacity={0.4} markerEnd="url(#arrowHead)" />
      </g>

      {CONV_BLOCKS.map((block, bi) => {
        const x = 142 + bi * 116;
        const accent = ["a", "b", "c", "d"][bi];
        return (
          <g key={block.label}>
            <FeatureMapStack x={x + 10} y={baseY + 34} w={block.w} h={50} maps={block.maps} delay={0.15 + bi * 0.12} accent={accent} />
            <g transform={`translate(${x + block.w / 2 + 10}, ${baseY - 6})`} opacity={0.55}>
              {Array.from({ length: 9 }).map((_, idx) => (
                <rect
                  key={idx}
                  x={(idx % 3) * 6 - 9}
                  y={Math.floor(idx / 3) * 6 - 9}
                  width={5}
                  height={5}
                  rx={1}
                  fill="var(--dps-primary)"
                  fillOpacity={0.2 + (idx % 4) * 0.1}
                  className="cnn-kernel-cell"
                  style={{ animationDelay: `${bi * 0.25 + idx * 0.04}s` }}
                />
              ))}
            </g>
            <text x={x + block.w / 2 + 10} y={baseY - 24} textAnchor="middle" className="cnn-label">{block.label}</text>
            <text x={x + block.w / 2 + 10} y={baseY - 10} textAnchor="middle" className="cnn-sublabel">
              {block.filters} filtres · k={block.kernel}
            </text>
            <text x={x + block.w / 2 + 10} y={baseY + 104} textAnchor="middle" className="cnn-micro">MaxPool ÷{block.pool}</text>
            {bi < CONV_BLOCKS.length - 1 && (
              <line
                x1={x + block.w + 22}
                y1={baseY + 18}
                x2={x + block.w + 38}
                y2={baseY + 18}
                stroke="var(--dps-primary)"
                strokeWidth={1.5}
                strokeOpacity={0.35}
                markerEnd="url(#arrowHead)"
                className="cnn-connector"
              />
            )}
          </g>
        );
      })}

      <g transform="translate(618, 0)">
        <path
          d={`M 4 ${baseY - 24} L 32 ${baseY + 52} L 60 ${baseY - 24} Z`}
          fill="var(--dps-primary-soft)"
          stroke="var(--dps-primary)"
          strokeWidth={1.2}
          strokeOpacity={0.55}
          filter="url(#cnnGlow)"
          className="cnn-gap"
        />
        <text x={32} y={baseY - 36} textAnchor="middle" className="cnn-label">GAP</text>
        <text x={32} y={baseY + 74} textAnchor="middle" className="cnn-sublabel">Dense 128</text>
        {Array.from({ length: 8 }).map((_, i) => (
          <circle
            key={i}
            cx={16 + (i % 4) * 12}
            cy={baseY + 90 + Math.floor(i / 4) * 12}
            r={3.5}
            fill="url(#nodePulse)"
            className="cnn-dense-node"
            style={{ animationDelay: `${1 + i * 0.08}s` }}
          />
        ))}
        <line x1={64} y1={baseY + 16} x2={88} y2={baseY + 16} stroke="var(--dps-primary)" strokeWidth={1.5} strokeOpacity={0.4} markerEnd="url(#arrowHead)" />
      </g>

      <g>
        <rect x={708} y={baseY - 82} width={188} height={168} rx={14} fill="var(--dps-surface)" stroke="var(--dps-border)" strokeWidth={1} opacity={0.95} />
        <text x={802} y={baseY - 62} textAnchor="middle" className="cnn-label">Sortie AASM</text>
        <text x={802} y={baseY - 48} textAnchor="middle" className="cnn-sublabel">Softmax · 5 classes</text>
        {stagePositions.map(({ stage, cx, cy }, i) => (
          <g key={stage}>
            <circle cx={cx} cy={cy} r={22} fill={STAGE_COLORS[stage]} fillOpacity={0.12} stroke={STAGE_COLORS[stage]} strokeWidth={1.8} className="cnn-output-ring" style={{ animationDelay: `${i * 0.35}s` }} />
            <circle cx={cx} cy={cy} r={7} fill={STAGE_COLORS[stage]} className="cnn-output-core" style={{ animationDelay: `${i * 0.35}s` }} />
            <text x={cx} y={cy + 34} textAnchor="middle" className="cnn-stage-label" fill={STAGE_COLORS[stage]}>{stage}</text>
          </g>
        ))}
      </g>

      <g transform="translate(340, 24)">
        <rect x={0} y={0} width={240} height={28} rx={14} fill="var(--dps-primary-soft)" stroke="var(--dps-primary)" strokeWidth={0.8} strokeOpacity={0.35} />
        <text x={120} y={18} textAnchor="middle" className="cnn-badge">CNN 1D · OpenVINO IR · NPU / GPU / CPU</text>
      </g>
    </svg>
  );
}

export function CnnArchitectureLoader() {
  const [phaseIdx, setPhaseIdx] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setPhaseIdx((p) => (p + 1) % PHASES.length);
    }, 2200);
    return () => clearInterval(id);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.45 }}
      className="cnn-loader mb-6 overflow-hidden rounded-2xl border border-[var(--dps-border-soft)] bg-[var(--dps-surface)] shadow-[var(--dps-shadow-lg)]"
    >
      <div className="cnn-loader-inner relative px-4 py-8 md:px-10 md:py-10">
        <div className="cnn-loader-grid pointer-events-none absolute inset-0 opacity-[0.35]" aria-hidden />

        <div className="relative z-10 flex flex-col items-center gap-6">
          <CnnArchitectureSvg />

          <div className="flex w-full max-w-lg flex-col items-center gap-3 text-center">
            <div className="flex items-center gap-2">
              <span className="cnn-status-dot" />
              <span className="text-sm font-medium text-[var(--dps-primary)]">Inférence en cours</span>
            </div>
            <AnimatePresence mode="wait">
              <motion.p
                key={phaseIdx}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.35 }}
                className="text-sm text-[var(--dps-text-muted)]"
              >
                {PHASES[phaseIdx]}
              </motion.p>
            </AnimatePresence>
            <div className="cnn-progress-track w-full max-w-xs">
              <div className="cnn-progress-bar" />
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
