import {MemoryOSHttpClient, type MemoryOSResponse} from "./client.js";
import {createHash} from "node:crypto";
export type SessionContext = {userId:string; projectId:string; sessionId:string; sessionKey?:string; prompt?:string; messages?:unknown[]};
export function stableSessionKey(ctx:SessionContext):string { return `session-${createHash("sha256").update([ctx.userId,ctx.projectId,"openclaw",ctx.sessionId].join("|")).digest("hex").slice(0,32)}`; }
export class MemoryOSContextEngine {
  private readonly turns = new Map<string,number>();
  constructor(private readonly client: MemoryOSHttpClient, private readonly tokenBudget = 1200, private readonly checkpointEvery = 10) {}
  async assemble(ctx: SessionContext): Promise<MemoryOSResponse> {
    return this.client.request("/v1/context/assemble", {query:ctx.prompt || ctx.projectId, user_id:ctx.userId, project_id:ctx.projectId, token_budget:this.tokenBudget, connect_metadata:{adapter_id:"openclaw"}});
  }
  async afterTurn(ctx: SessionContext) { const key=stableSessionKey(ctx); const result=await this.append(ctx,"TURN_END"); const count=(this.turns.get(key) ?? 0)+1; this.turns.set(key,count); if (count % this.checkpointEvery===0) await this.client.request(`/v1/sessions/${encodeURIComponent(key)}/checkpoint`,{}); return result; }
  async session_start(ctx: SessionContext) { await this.append(ctx, "SESSION_START"); return this.assemble(ctx); }
  async compact(ctx: SessionContext) { const key=stableSessionKey(ctx); await this.append(ctx, "PRE_COMPACT"); return this.client.request(`/v1/sessions/${encodeURIComponent(key)}/checkpoint`, {}); }
  async session_end(ctx: SessionContext) { const key=stableSessionKey(ctx); await this.append(ctx, "SESSION_END"); return this.client.request(`/v1/sessions/${encodeURIComponent(key)}/finalize`, {}); }
  async before_reset(ctx: SessionContext) { return this.session_end(ctx); }
  private append(ctx: SessionContext, eventType: string) { return this.client.request("/v1/sessions/events", {event_id:crypto.randomUUID(), event_type:eventType, adapter_id:"openclaw", user_id:ctx.userId, project_id:ctx.projectId, session_id:ctx.sessionId, session_key:stableSessionKey(ctx), prompt:ctx.prompt, messages:ctx.messages ?? []}); }
}
