import {MemoryOSHttpClient} from "./client.js";
export function memoryTools(client: MemoryOSHttpClient) {
  return {
    memory_recall: (args:Record<string,unknown>) => client.request("/v1/context/search", args),
    memory_read: (args:Record<string,unknown>) => client.request(`/v1/context/read?uri=${encodeURIComponent(String(args.uri ?? ""))}`, undefined, "GET"),
    memoryos_status: () => client.health(),
    memoryos_recall_trace: (args:Record<string,unknown>) => client.request(`/v1/recall-traces/${encodeURIComponent(String(args.trace_id ?? ""))}`, undefined, "GET"),
  };
}
