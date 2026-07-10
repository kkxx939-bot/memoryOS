import {MemoryOSHttpClient} from "./client.js";
export function readRecallTrace(client:MemoryOSHttpClient, traceId:string) { return client.request(`/v1/recall-traces/${encodeURIComponent(traceId)}`,undefined,"GET"); }
