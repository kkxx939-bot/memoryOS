import {MemoryOSHttpClient} from "./client.js";
import {MemoryOSContextEngine, stableSessionKey, type SessionContext} from "./context-engine.js";
import {normalizeConfig} from "./config.js";
import {memoryTools} from "./tools.js";
import {createHash} from "node:crypto";
import {execFileSync} from "node:child_process";
import {basename} from "node:path";

export {MemoryOSHttpClient} from "./client.js";
export {MemoryOSContextEngine, stableSessionKey} from "./context-engine.js";
export {memoryTools} from "./tools.js";
export {normalizeConfig} from "./config.js";
export {readRecallTrace} from "./recall-trace.js";

type ToolDefinition = {
  name:string;
  label:string;
  description:string;
  parameters:Record<string,unknown>;
  execute:(toolCallId:string,args:Record<string,unknown>)=>Promise<unknown>;
};
type OpenClawPluginApi = {
  pluginConfig?:unknown;
  logger:{info:(message:string)=>void;warn:(message:string)=>void;error:(message:string)=>void};
  registerTool:(tool:ToolDefinition,options?:{name?:string})=>void;
  registerContextEngine?:(id:string,factory:()=>unknown)=>void;
  on:(hookName:string,handler:(event:unknown,context?:unknown)=>unknown,options?:{priority?:number})=>void;
};

const plugin = {
  id:"memoryos",
  name:"MemoryOS",
  description:"MemoryOS context engine with remote recall and session synchronization",
  kind:"context-engine" as const,
  register(api:OpenClawPluginApi) {
    const raw = (api.pluginConfig && typeof api.pluginConfig === "object" ? api.pluginConfig : {}) as Record<string,unknown>;
    const tokenEnv = String(raw.apiTokenEnv ?? "MEMORYOS_API_TOKEN");
    const cfg = normalizeConfig({
      baseUrl:String(raw.baseUrl ?? "http://127.0.0.1:8765"),
      apiToken:typeof raw.apiToken === "string" ? raw.apiToken : process.env[tokenEnv],
      timeoutMs:typeof raw.timeoutMs === "number" ? raw.timeoutMs : undefined,
      tokenBudget:typeof raw.tokenBudget === "number" ? raw.tokenBudget : undefined,
    });
    const client = new MemoryOSHttpClient(cfg.baseUrl,cfg.apiToken,cfg.timeoutMs);
    const engine = new MemoryOSContextEngine(client,cfg.tokenBudget);
    const defaultUserId=String(raw.userId ?? process.env.MEMORYOS_USER_ID ?? "default");
    const defaultProjectId=String(raw.projectId ?? process.env.MEMORYOS_PROJECT_ID ?? projectIdentity(process.cwd()));
    api.registerContextEngine?.("memoryos",()=>engine);
    for (const [name,execute] of Object.entries(memoryTools(client))) {
      api.registerTool({name,label:name,description:`MemoryOS ${name}`,parameters:{type:"object",additionalProperties:true},execute:async (_id,args)=>execute(args)}, {name});
    }
    api.on("session_start",(event,context)=>engine.session_start(toSessionContext(event,context,defaultUserId,defaultProjectId)));
    api.on("session_end",(event,context)=>engine.session_end(toSessionContext(event,context,defaultUserId,defaultProjectId)));
    api.on("before_reset",(event,context)=>engine.before_reset(toSessionContext(event,context,defaultUserId,defaultProjectId)));
    api.logger.info("memoryos: remote context engine registered");
  },
};

function toSessionContext(event:unknown,context:unknown,defaultUserId:string,defaultProjectId:string):SessionContext {
  const payload = (event && typeof event === "object" ? event : {}) as Record<string,unknown>;
  const ctx = (context && typeof context === "object" ? context : {}) as Record<string,unknown>;
  return {
    userId:String(payload.userId ?? payload.user_id ?? ctx.userId ?? ctx.user_id ?? defaultUserId),
    projectId:String(payload.projectId ?? payload.project_id ?? ctx.projectId ?? ctx.project_id ?? defaultProjectId),
    sessionId:String(payload.sessionId ?? payload.session_id ?? ctx.sessionId ?? ctx.session_id ?? "unknown"),
    sessionKey:typeof ctx.sessionKey === "string" ? ctx.sessionKey : undefined,
    prompt:typeof payload.prompt === "string" ? payload.prompt : undefined,
    messages:Array.isArray(payload.messages) ? payload.messages : [],
  };
}

function projectIdentity(directory:string):string {
  let identity="";
  try {identity=execFileSync("git",["remote","get-url","origin"],{cwd:directory,encoding:"utf8",timeout:1000}).trim().toLowerCase().replace(/\.git$/,"").replace(/^git@([^:]+):/,"$1/").replace(/^https?:\/\//,"").replace(/\/$/,"");}
  catch {identity=`local-repository:${basename(directory).toLowerCase()}`;}
  return `project-${createHash("sha256").update(identity).digest("hex").slice(0,24)}`;
}

export default plugin;
