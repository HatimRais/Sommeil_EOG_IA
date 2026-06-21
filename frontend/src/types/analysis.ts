export interface PatientInfo {
  id: string;
  recordingDate: string;
  durationMin: number;
  durationFormatted: string;
  sourceFile: string;
  eogChannel: string;
  sfreqOrig: number;
  sfreqTarget: number;
  nEpochs: number;
  inferenceMs: number;
}

export interface Prediction {
  epoch: number;
  stage: string;
  stageIndex: number;
  confidence: number;
  timeHours: number;
}

export interface ClinicalReport {
  TIB_min: number;
  TST_min: number;
  SE_pct: number;
  Latency_min: number | null;
  WASO_min: number;
  REM_lat_min: number | null;
  Awakenings: number;
  Transitions: number;
  Fragmentation_idx: number;
  Cycles: number;
  Counts: Record<string, number>;
  W_pct: number;
  N1_pct: number;
  N2_pct: number;
  N3_pct: number;
  REM_pct: number;
}

export interface MetricCard {
  label: string;
  value: string;
  raw: number | null;
  status: "norm" | "warn" | "alert" | "info";
  statusLabel: string;
  reference: string;
}

export interface InterpretationItem {
  text: string;
  severity: "norm" | "warn" | "alert";
}

export interface ValidationResult {
  accuracy?: number;
  kappa?: number | null;
  kappaLabel?: string;
  epochsEvaluated?: number;
  perStage?: Array<{
    stage: string;
    sensitivity: number;
    precision: number;
    f1: number;
    support: number;
  }>;
  confusion?: {
    matrix: number[][];
    matrixNorm: number[][];
    stages: string[];
  };
  expertStages?: (string | null)[];
  parseError?: boolean;
}

export interface AnalysisResult {
  patient: PatientInfo;
  predictions: Prediction[];
  clinical: ClinicalReport;
  metrics: MetricCard[];
  interpretation: InterpretationItem[];
  validation: ValidationResult | null;
  engine: {
    device: string;
    runtimeDevice: string;
    hardware: string | null;
    inferenceMs: number;
    throughput: number;
    meanConfidence: number;
  };
  technical: Record<string, string | number>;
  stages: { short: string[]; full: string[] };
  cycles: number;
  meanConfidence: number;
  fragmentation: number;
  transitions: number;
}

export interface DeviceOption {
  id: string;
  label: string;
}

export interface PatientRecord {
  id: string;
  signalFile: string;
  hasLabels: boolean;
}
