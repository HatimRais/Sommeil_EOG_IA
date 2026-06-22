import { proxyGet } from "@/lib/inference-proxy";

export async function GET() {
  return proxyGet("/api/devices");
}

export const runtime = "nodejs";
