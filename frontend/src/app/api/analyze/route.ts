import { proxyPostForm } from "@/lib/inference-proxy";

export async function POST(request: Request) {
  const formData = await request.formData();
  return proxyPostForm("/api/analyze", formData);
}

export const runtime = "nodejs";
export const maxDuration = 300;
