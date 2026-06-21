import type { AnalysisResult, DeviceOption, PatientRecord } from "@/types/analysis";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || `Erreur API (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export async function fetchDevices(): Promise<DeviceOption[]> {
  const data = await handleResponse<{ options: DeviceOption[] }>(
    await fetch(`${API_BASE}/api/devices`)
  );
  return data.options;
}

export async function fetchPatients(): Promise<PatientRecord[]> {
  const data = await handleResponse<{ patients: PatientRecord[] }>(
    await fetch(`${API_BASE}/api/patients`)
  );
  return data.patients;
}

export async function analyzeRecording(params: {
  device: string;
  signal?: File;
  labels?: File;
  patientId?: string;
}): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("device", params.device);
  if (params.patientId) {
    form.append("patient_id", params.patientId);
  }
  if (params.signal) {
    form.append("signal", params.signal);
  }
  if (params.labels) {
    form.append("labels", params.labels);
  }

  return handleResponse<AnalysisResult>(
    await fetch(`${API_BASE}/api/analyze`, { method: "POST", body: form })
  );
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}
