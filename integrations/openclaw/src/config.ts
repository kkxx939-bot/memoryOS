export type MemoryOSPluginConfig = {baseUrl:string; timeoutMs?:number};
export function normalizeConfig(value: Partial<MemoryOSPluginConfig>): Required<MemoryOSPluginConfig> {
  if (!value.baseUrl?.startsWith("http://") && !value.baseUrl?.startsWith("https://")) throw new Error("MemoryOS baseUrl must use HTTP or HTTPS");
  return {baseUrl:value.baseUrl, timeoutMs:value.timeoutMs ?? 2500};
}
