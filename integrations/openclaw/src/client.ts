export type MemoryOSResponse = Record<string, unknown>;
export class MemoryOSHttpClient {
  constructor(readonly baseUrl: string, readonly token?: string, readonly timeoutMs = 2500) {}
  async request(path: string, payload?: unknown, method = "POST"): Promise<MemoryOSResponse> {
    const headers: Record<string,string> = {"content-type":"application/json", "x-request-id": crypto.randomUUID()};
    if (this.token) headers.authorization = `Bearer ${this.token}`;
    try {
      const response = await fetch(`${this.baseUrl.replace(/\/$/, "")}${path}`, {method, headers, body: method === "GET" ? undefined : JSON.stringify(payload ?? {}), signal: AbortSignal.timeout(this.timeoutMs)});
      const body = await response.json() as MemoryOSResponse;
      return response.ok ? body : {error: body.error ?? {code:"HTTP_ERROR", message:`HTTP ${response.status}`, retryable: response.status >= 500}};
    } catch (error) {
      return {error:{code:"REMOTE_UNAVAILABLE", message:error instanceof Error ? error.message : "request failed", retryable:true}};
    }
  }
  health() { return this.request("/health", undefined, "GET"); }
}
