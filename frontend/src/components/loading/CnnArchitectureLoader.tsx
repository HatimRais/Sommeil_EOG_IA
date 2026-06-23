"use client";

import { STAGE_COLORS } from "@/lib/theme";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";

const PHASES = [
  { text: "Lecture du signal EOG…", section: "input" as const },
  { text: "Prétraitement · filtre 0,5–35 Hz · z-score", section: "preprocess" as const },
  { text: "Extraction des motifs convolutifs (4 blocs)", section: "conv" as const },
  { text: "Inférence OpenVINO · batch 64 époques", section: "inference" as const },
  { text: "Classification AASM · 5 stades", section: "classify" as const },
  { text: "Construction de l'hypnogramme…", section: "hypno" as const },
];

const CONV_BLOCKS = [
  { label: "Conv₁", filters: 64, kernel: 11, pool: 4, maps: 4, w: 88, out: "(750, 64)" },
  { label: "Conv₂", filters: 128, kernel: 7, pool: 4, maps: 5, w: 62, out: "(187, 128)" },
  { label: "Conv₃", filters: 256, kernel: 5, pool: 4, maps: 6, w: 44, out: "(46, 256)" },
  { label: "Conv₄", filters: 256, kernel: 3, pool: 2, maps: 6, w: 30, out: "(23, 256)" },
];

const STAGES = ["W", "N1", "N2", "N3", "REM"] as const;

type Section = (typeof PHASES)[number]["section"];

function sectionActive(active: Section, target: Section | Section[]) {
  const targets = Array.isArray(target) ? target : [target];
  return targets.includes(active);
}

function FeatureMapStack({
  x,
  y,
  w,
  h,
  maps,
  delay,
  accent,
  lit,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  maps: number;
  delay: number;
  accent: string;
  lit: boolean;
}) {
  return (
    <g opacity={lit ? 1 : 0.72}>
      {Array.from({ length: maps }).map((_, i) => (
        <motion.rect
          key={i}
          x={x + i * 5}
          y={y - i * 6}
          width={w}
          height={h}
          rx={4}
          fill={`url(#fmGrad-${accent})`}
          stroke={lit ? "var(--dps-accent)" : "var(--dps-primary)"}
          strokeWidth={lit ? 1.2 : 0.6}
          strokeOpacity={lit ? 0.7 : 0.35}
          initial={{ opacity: 0, scale: 0.85 }}
          animate={{ opacity: lit ? 0.75 + i * 0.04 : 0.5 + i * 0.06, scale: 1 }}
          transition={{ delay: delay + i * 0.08, duration: 0.5 }}
          className="cnn-feature-map"
        />
      ))}
    </g>
  );
}

function CnnArchitectureSvg({ activeSection, activeConvBlock }: { activeSection: Section; activeConvBlock: number }) {
  const vbW = 920;
  const vbH = 380;
  const baseY = 210;

  const stagePositions: { stage: (typeof STAGES)[number]; cx: number; cy: number }[] = [
    { stage: "W", cx: 748, cy: baseY - 30 },
    { stage: "N1", cx: 808, cy: baseY - 30 },
    { stage: "N2", cx: 868, cy: baseY - 30 },
    { stage: "N3", cx: 778, cy: baseY + 38 },
    { stage: "REM", cx: 838, cy: baseY + 38 },
  ];

  const softmaxHeights = [0.55, 0.35, 0.85, 0.7, 0.5];

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
        <linearGradient id="scanGrad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="var(--dps-accent)" stopOpacity="0" />
          <stop offset="50%" stopColor="var(--dps-accent)" stopOpacity="0.85" />
          <stop offset="100%" stopColor="var(--dps-accent)" stopOpacity="0" />
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
        <filter id="sectionGlow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="6" result="blur" />
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
        <clipPath id="eogClip">
          <rect x={26} y={baseY - 68} width={84} height={136} rx={8} />
        </clipPath>
      </defs>

      <ellipse cx={460} cy={178} rx={420} ry={130} fill="url(#cnnBgGlow)" className="cnn-ambient" />

      {/* Flux principal animé */}
      <path
        d="M 118 210 C 210 208, 280 212, 360 210 S 540 208, 640 210 S 740 212, 820 210"
        fill="none"
        stroke="var(--dps-primary)"
        strokeWidth={1.5}
        strokeOpacity={0.12}
        strokeDasharray="8 12"
        className="cnn-backbone"
      />

      {[0, 1, 2, 3, 4, 5, 6].map((i) => (
        <circle key={i} r={i % 2 === 0 ? 3.5 : 2} fill="var(--dps-accent)" className="cnn-particle">
          <animateMotion
            dur={`${2.2 + i * 0.2}s`}
            repeatCount="indefinite"
            begin={`${i * 0.35}s`}
            path="M 118 210 C 210 208, 280 212, 360 210 S 540 208, 640 210 S 740 212, 820 210"
          />
          <animate attributeName="opacity" values="0.3;1;0.3" dur={`${2.2 + i * 0.2}s`} repeatCount="indefinite" begin={`${i * 0.35}s`} />
        </circle>
      ))}

      {/* ── Entrée EOG ── */}
      <g filter={sectionActive(activeSection, ["input", "preprocess"]) ? "url(#sectionGlow)" : undefined}>
        <rect
          x={18}
          y={baseY - 76}
          width={100}
          height={152}
          rx={12}
          fill="var(--dps-surface)"
          stroke={sectionActive(activeSection, ["input", "preprocess"]) ? "var(--dps-accent)" : "var(--dps-border)"}
          strokeWidth={sectionActive(activeSection, ["input", "preprocess"]) ? 1.8 : 1}
          opacity={0.95}
          className={sectionActive(activeSection, "input") ? "cnn-section-pulse" : undefined}
        />
        <text x={68} y={baseY - 86} textAnchor="middle" className="cnn-label">
          Entrée EOG
        </text>
        <text x={68} y={baseY - 70} textAnchor="middle" className="cnn-sublabel">
          3000 × 1 · 100 Hz
        </text>
        <g clipPath="url(#eogClip)">
          <path
            d="M 30 208 C 38 178, 46 208, 54 208 S 70 188, 78 218 S 90 198, 102 212"
            fill="none"
            stroke="url(#waveGrad)"
            strokeWidth={2.4}
            strokeLinecap="round"
            className="cnn-wave-path"
          />
          <path
            d="M 30 212 C 42 242, 54 212, 66 212 S 82 232, 94 202 S 106 222, 110 210"
            fill="none"
            stroke="url(#waveGrad)"
            strokeWidth={1.2}
            strokeOpacity={0.35}
            strokeLinecap="round"
            className="cnn-wave-path-alt"
          />
          <rect x={28} y={baseY - 66} width={8} height={132} fill="url(#scanGrad)" className="cnn-scan-line" />
        </g>
        {sectionActive(activeSection, "preprocess") && (
          <g className="cnn-filter-badge">
            <rect x={24} y={baseY + 58} width={88} height={16} rx={8} fill="var(--dps-primary-soft)" stroke="var(--dps-accent)" strokeWidth={0.6} />
            <text x={68} y={baseY + 69} textAnchor="middle" className="cnn-micro">
              0.5–35 Hz · z-score
            </text>
          </g>
        )}
        <line x1={118} y1={baseY + 18} x2={138} y2={baseY + 18} stroke="var(--dps-primary)" strokeWidth={1.5} strokeOpacity={0.4} markerEnd="url(#arrowHead)" className="cnn-connector" />
      </g>

      {/* ── 4 blocs Conv ── */}
      {CONV_BLOCKS.map((block, bi) => {
        const x = 142 + bi * 116;
        const accent = ["a", "b", "c", "d"][bi];
        const lit = sectionActive(activeSection, ["conv", "inference"]);
        return (
          <g key={block.label} filter={lit && bi === activeConvBlock ? "url(#sectionGlow)" : undefined}>
            <FeatureMapStack
              x={x + 10}
              y={baseY + 34}
              w={block.w}
              h={50}
              maps={block.maps}
              delay={0.15 + bi * 0.12}
              accent={accent}
              lit={lit}
            />
            <g transform={`translate(${x + block.w / 2 + 10}, ${baseY - 6})`} opacity={lit ? 0.85 : 0.55}>
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
            <text x={x + block.w / 2 + 10} y={baseY - 24} textAnchor="middle" className="cnn-label">
              {block.label}
            </text>
            <text x={x + block.w / 2 + 10} y={baseY - 10} textAnchor="middle" className="cnn-sublabel">
              {block.filters} filtres · k={block.kernel}
            </text>
            <text x={x + block.w / 2 + 10} y={baseY + 104} textAnchor="middle" className="cnn-micro">
              BN → ReLU → MaxPool ÷{block.pool}
            </text>
            <text x={x + block.w / 2 + 10} y={baseY + 116} textAnchor="middle" className="cnn-micro" opacity={0.6}>
              {block.out}
            </text>
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

      {/* ── GAP + Dense ── */}
      <g transform="translate(618, 0)" filter={sectionActive(activeSection, ["inference", "classify"]) ? "url(#sectionGlow)" : undefined}>
        <path
          d={`M 4 ${baseY - 24} L 32 ${baseY + 52} L 60 ${baseY - 24} Z`}
          fill="var(--dps-primary-soft)"
          stroke="var(--dps-primary)"
          strokeWidth={1.2}
          strokeOpacity={0.55}
          filter="url(#cnnGlow)"
          className="cnn-gap"
        />
        <text x={32} y={baseY - 36} textAnchor="middle" className="cnn-label">
          GAP
        </text>
        <text x={32} y={baseY + 74} textAnchor="middle" className="cnn-sublabel">
          GlobalAvgPool
        </text>
        <text x={32} y={baseY + 88} textAnchor="middle" className="cnn-micro">
          Dense 128 → 5
        </text>
        {Array.from({ length: 8 }).map((_, i) => (
          <circle
            key={i}
            cx={16 + (i % 4) * 12}
            cy={baseY + 100 + Math.floor(i / 4) * 12}
            r={3.5}
            fill="url(#nodePulse)"
            className="cnn-dense-node"
            style={{ animationDelay: `${1 + i * 0.08}s` }}
          />
        ))}
        <line x1={64} y1={baseY + 16} x2={88} y2={baseY + 16} stroke="var(--dps-primary)" strokeWidth={1.5} strokeOpacity={0.4} markerEnd="url(#arrowHead)" className="cnn-connector" />
      </g>

      {/* ── Sortie AASM + softmax ── */}
      <g filter={sectionActive(activeSection, ["classify", "hypno"]) ? "url(#sectionGlow)" : undefined}>
        <rect
          x={708}
          y={baseY - 82}
          width={188}
          height={168}
          rx={14}
          fill="var(--dps-surface)"
          stroke={sectionActive(activeSection, ["classify", "hypno"]) ? "var(--dps-accent)" : "var(--dps-border)"}
          strokeWidth={sectionActive(activeSection, ["classify", "hypno"]) ? 1.8 : 1}
          opacity={0.95}
        />
        <text x={802} y={baseY - 62} textAnchor="middle" className="cnn-label">
          Sortie AASM
        </text>
        <text x={802} y={baseY - 48} textAnchor="middle" className="cnn-sublabel">
          Softmax · 5 classes
        </text>
        {stagePositions.map(({ stage, cx, cy }, i) => (
          <g key={stage}>
            <circle
              cx={cx}
              cy={cy}
              r={22}
              fill={STAGE_COLORS[stage]}
              fillOpacity={0.12}
              stroke={STAGE_COLORS[stage]}
              strokeWidth={1.8}
              className="cnn-output-ring"
              style={{ animationDelay: `${i * 0.35}s` }}
            />
            <circle
              cx={cx}
              cy={cy}
              r={7}
              fill={STAGE_COLORS[stage]}
              className="cnn-output-core"
              style={{ animationDelay: `${i * 0.35}s` }}
            />
            <text x={cx} y={cy + 34} textAnchor="middle" className="cnn-stage-label" fill={STAGE_COLORS[stage]}>
              {stage}
            </text>
            {sectionActive(activeSection, ["classify", "hypno"]) && (
              <rect
                x={cx - 8}
                y={baseY + 80 - 28 * softmaxHeights[i]}
                width={16}
                height={28 * softmaxHeights[i]}
                rx={2}
                fill={STAGE_COLORS[stage]}
                fillOpacity={0.55}
                className="cnn-softmax-bar"
                style={{ animationDelay: `${i * 0.15}s` }}
              />
            )}
          </g>
        ))}
      </g>

      {/* Badge architecture */}
      <g transform="translate(340, 24)">
        <rect x={0} y={0} width={240} height={28} rx={14} fill="var(--dps-primary-soft)" stroke="var(--dps-primary)" strokeWidth={0.8} strokeOpacity={0.35} />
        <text x={120} y={18} textAnchor="middle" className="cnn-badge">
          CNN 1D · 455k params · OpenVINO IR
        </text>
      </g>

      {/* Batch inférence */}
      <g transform="translate(340, 348)" className={sectionActive(activeSection, "inference") ? "cnn-batch-active" : undefined}>
        <rect x={0} y={0} width={240} height={22} rx={11} fill="var(--dps-surface-2)" stroke="var(--dps-border)" strokeWidth={0.8} />
        <rect x={2} y={2} width={120} height={18} rx={9} fill="var(--dps-primary)" fillOpacity={0.15} className="cnn-batch-fill" />
        <text x={120} y={15} textAnchor="middle" className="cnn-micro">
          Batch 64 époques · 30 s @ 100 Hz
        </text>
      </g>
    </svg>
  );
}

export function CnnArchitectureLoader() {
  const [phaseIdx, setPhaseIdx] = useState(0);
  const [batchTick, setBatchTick] = useState(0);
  const [convBlock, setConvBlock] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setPhaseIdx((p) => (p + 1) % PHASES.length);
    }, 2200);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const id = setInterval(() => {
      setBatchTick((t) => (t >= 64 ? 0 : t + 4));
    }, 180);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const id = setInterval(() => {
      setConvBlock((b) => (b + 1) % CONV_BLOCKS.length);
    }, 550);
    return () => clearInterval(id);
  }, []);

  const phase = PHASES[phaseIdx];
  const activeConvBlock =
    phase.section === "conv" ? convBlock : phase.section === "inference" ? CONV_BLOCKS.length - 1 : 0;

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
          <CnnArchitectureSvg activeSection={phase.section} activeConvBlock={activeConvBlock} />

          {/* Indicateurs de phase */}
          <div className="flex flex-wrap justify-center gap-1.5">
            {PHASES.map((p, i) => (
              <span
                key={p.section}
                className={`cnn-phase-dot h-1.5 rounded-full transition-all duration-300 ${
                  i === phaseIdx ? "cnn-phase-dot-active w-6 bg-[var(--dps-primary)]" : "w-1.5 bg-[var(--dps-border)]"
                }`}
              />
            ))}
          </div>

          <div className="flex w-full max-w-lg flex-col items-center gap-3 text-center">
            <div className="flex items-center gap-2">
              <span className="cnn-status-dot" />
              <span className="text-sm font-medium text-[var(--dps-primary)]">
                Inférence en cours
                {phase.section === "inference" && (
                  <span className="ml-2 font-mono text-xs text-[var(--dps-text-muted)]">
                    [{batchTick}/64]
                  </span>
                )}
              </span>
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
                {phase.text}
              </motion.p>
            </AnimatePresence>
            <div className="cnn-progress-track w-full max-w-xs">
              <div
                className="cnn-progress-bar"
                style={{ width: `${((phaseIdx + 1) / PHASES.length) * 100}%`, animation: "none", transform: "none" }}
              />
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
