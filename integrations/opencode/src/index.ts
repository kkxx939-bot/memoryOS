import {createHash} from "node:crypto";
import {execFileSync} from "node:child_process";
import {basename} from "node:path";

export type MemoryRequest = (path:string, body?:unknown, method?:string) => Promise<Record<string,unknown>>;
export function createMemoryTools(request: MemoryRequest) {
  return {
    memory_recall:(args:Record<string,unknown>)=>request("/v1/context/search",args),
    memory_store:(args:Record<string,unknown>)=>request("/v1/memories/remember",args),
    memory_forget:(args:Record<string,unknown>)=>request("/v1/memories/forget",args),
    memory_read:(args:Record<string,unknown>)=>request(`/v1/context/read?uri=${encodeURIComponent(String(args.uri ?? ""))}`,undefined,"GET"),
  };
}
export function createSessionSync(request: MemoryRequest) {
  const append=(event_type:string,args:Record<string,unknown>)=>request("/v1/sessions/events",{...args,event_type,adapter_id:"opencode"});
  return {append, afterTurn:(args:Record<string,unknown>)=>append("TURN_END",args), checkpoint:(sessionKey:string)=>request(`/v1/sessions/${encodeURIComponent(sessionKey)}/checkpoint`,{}), finalize:(sessionKey:string)=>request(`/v1/sessions/${encodeURIComponent(sessionKey)}/finalize`,{})};
}

export async function MemoryOSOpenCodePlugin({directory}:{directory:string}) {
  const baseUrl=(process.env.MEMORYOS_BASE_URL ?? "http://127.0.0.1:8765").replace(/\/$/,"");
  const timeoutMs=Number(process.env.MEMORYOS_TIMEOUT_MS ?? 2500);
  const token=process.env.MEMORYOS_API_TOKEN;
  const userId=process.env.MEMORYOS_USER_ID ?? "default";
  const projectId=process.env.MEMORYOS_PROJECT_ID ?? projectIdentity(directory);
  const request:MemoryRequest=async(path,body,method="POST")=>{
    const headers:Record<string,string>={"content-type":"application/json","x-request-id":crypto.randomUUID()};
    if(token) headers.authorization=`Bearer ${token}`;
    try {
      const response=await fetch(`${baseUrl}${path}`,{method,headers,body:method==="GET"?undefined:JSON.stringify(body ?? {}),signal:AbortSignal.timeout(timeoutMs)});
      const payload=await response.json() as Record<string,unknown>;
      return response.ok?payload:{error:payload.error ?? {code:"HTTP_ERROR",message:`HTTP ${response.status}`,retryable:response.status>=500}};
    } catch(error) {
      return {error:{code:"REMOTE_UNAVAILABLE",message:error instanceof Error?error.message:"request failed",retryable:true}};
    }
  };
  const rawTools=createMemoryTools(request);
  const sync=createSessionSync(request);
  const activeSessions=new Set<string>();
  const sessionKey=(nativeId:string)=>`session-${createHash("sha256").update([userId,projectId,"opencode",nativeId].join("|")).digest("hex").slice(0,32)}`;
  const eventPayload=(nativeId:string,extra:Record<string,unknown>={})=>({event_id:crypto.randomUUID(),user_id:userId,project_id:projectId,session_id:nativeId,session_key:sessionKey(nativeId),...extra});
  const finalize=async(nativeId:string)=>{if(!nativeId)return;await sync.finalize(sessionKey(nativeId));activeSessions.delete(nativeId);};
  return {
    event:async({event}:{event?:Record<string,unknown>})=>{
      const nativeId=resolveSessionId(event);
      if(!nativeId)return;
      if(event?.type==="session.created") {activeSessions.add(nativeId);await sync.append("SESSION_START",eventPayload(nativeId));}
      else if(event?.type==="message.updated") {
        const info=(event.properties as Record<string,unknown> | undefined)?.info as Record<string,unknown> | undefined;
        if(info?.role==="user" || (info?.role==="assistant" && info?.finish==="stop")) {await sync.afterTurn(eventPayload(nativeId,{messages:[info]}));await sync.checkpoint(sessionKey(nativeId));}
      }
      else if(event?.type==="session.compacted") {await sync.append("PRE_COMPACT",eventPayload(nativeId));await sync.checkpoint(sessionKey(nativeId));}
      else if(event?.type==="session.deleted" || event?.type==="session.error") await finalize(nativeId);
    },
    tool:{
      memory_recall:defineTool("Search MemoryOS context",{query:{type:"string"}},(args)=>rawTools.memory_recall(args)),
      memory_store:defineTool("Store an explicit MemoryOS memory",{content:{type:"string"},memory_type:{type:"string",optional:true},project_id:{type:"string",optional:true}},(args)=>rawTools.memory_store({user_id:userId,project_id:args.project_id ?? projectId,...args})),
      memory_forget:defineTool("Forget an exact MemoryOS URI",{uri:{type:"string"}},(args)=>rawTools.memory_forget({user_id:userId,...args})),
      memory_read:defineTool("Read an exact MemoryOS URI",{uri:{type:"string"}},(args)=>rawTools.memory_read(args)),
    },
    "experimental.session.compacting":async(input:{sessionID:string})=>{await sync.append("PRE_COMPACT",eventPayload(input.sessionID));return sync.checkpoint(sessionKey(input.sessionID));},
    stop:async()=>{for(const nativeId of [...activeSessions])await finalize(nativeId);},
  };
}

function defineTool(description:string,args:Record<string,unknown>,execute:(args:Record<string,unknown>,context?:unknown)=>Promise<unknown>) {
  return {description,args,execute};
}

function resolveSessionId(event?:Record<string,unknown>):string {
  const properties=event?.properties as Record<string,unknown> | undefined;
  const info=properties?.info as Record<string,unknown> | undefined;
  const infoId=event?.type==="session.created"?info?.id:undefined;
  return String(info?.sessionID ?? infoId ?? properties?.sessionID ?? properties?.sessionId ?? "");
}

export default MemoryOSOpenCodePlugin;

function projectIdentity(directory:string):string {
  let identity="";
  try {identity=execFileSync("git",["remote","get-url","origin"],{cwd:directory,encoding:"utf8",timeout:1000}).trim().toLowerCase().replace(/\.git$/,"").replace(/^git@([^:]+):/,"$1/").replace(/^https?:\/\//,"").replace(/\/$/,"");}
  catch {identity=`local-repository:${basename(directory).toLowerCase()}`;}
  return `project-${createHash("sha256").update(identity).digest("hex").slice(0,24)}`;
}
