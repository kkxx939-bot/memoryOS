export type MemoryOSPluginConfig = {baseUrl:string; apiToken?:string; timeoutMs?:number; tokenBudget?:number};
export function normalizeConfig(value: Partial<MemoryOSPluginConfig>): Required<Omit<MemoryOSPluginConfig,"apiToken">> & {apiToken?:string} {
  if (!value.baseUrl?.startsWith("http://") && !value.baseUrl?.startsWith("https://")) throw new Error("MemoryOS baseUrl must use HTTP or HTTPS");
  return {baseUrl:value.baseUrl, apiToken:value.apiToken, timeoutMs:value.timeoutMs ?? 2500, tokenBudget:value.tokenBudget ?? 1200};
}
