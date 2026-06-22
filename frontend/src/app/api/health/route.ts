import { proxyGet } from "@/lib/inference-proxy";

export async function GET() {
  return proxyGet("/api/health");
}

export const runtime = "nodejs";
