export function inferenceBaseUrl(): string {
  return (
    process.env.DEEPSLEEP_INFERENCE_URL?.replace(/\/$/, "") ||
    process.env.INFERENCE_API_URL?.replace(/\/$/, "") ||
    "http://127.0.0.1:8000"
  );
}

export async function proxyGet(path: string): Promise<Response> {
  try {
    const res = await fetch(`${inferenceBaseUrl()}${path}`, { cache: "no-store" });
    const body = await res.text();
    return new Response(body, {
      status: res.status,
      headers: { "Content-Type": res.headers.get("Content-Type") ?? "application/json" },
    });
  } catch {
    return new Response(JSON.stringify({ error: "API d'inférence injoignable." }), {
      status: 503,
      headers: { "Content-Type": "application/json" },
    });
  }
}

export async function proxyPostForm(path: string, formData: FormData): Promise<Response> {
  try {
    const res = await fetch(`${inferenceBaseUrl()}${path}`, {
      method: "POST",
      body: formData,
    });
    const body = await res.text();
    return new Response(body, {
      status: res.status,
      headers: { "Content-Type": res.headers.get("Content-Type") ?? "application/json" },
    });
  } catch {
    return new Response(
      JSON.stringify({
        detail: "Impossible de joindre l'API d'inférence.",
        error: "Impossible de joindre l'API d'inférence.",
      }),
      { status: 503, headers: { "Content-Type": "application/json" } },
    );
  }
}
