import { proxyGet } from "@/lib/inference-proxy";

export async function GET() {
  return proxyGet("/api/patients");
}

export const runtime = "nodejs";
