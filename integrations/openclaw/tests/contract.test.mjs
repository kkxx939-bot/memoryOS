import assert from "node:assert/strict";
import {createServer} from "node:http";
import test from "node:test";
import memoryOSPlugin, {MemoryOSContextEngine, MemoryOSHttpClient, stableSessionKey} from "../dist/index.js";
test("OpenClaw lifecycle uses MemoryOS HTTP contract", async () => {
  const calls=[]; const server=createServer((req,res)=>{let body="";req.on("data",c=>body+=c);req.on("end",()=>{calls.push({url:req.url,body});res.writeHead(200,{"content-type":"application/json"});res.end(JSON.stringify({status:"ok",packed_context:"ctx"}));});});
  await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve)); const port=server.address().port;
  const engine=new MemoryOSContextEngine(new MemoryOSHttpClient(`http://127.0.0.1:${port}`)); const ctx={userId:"u1",projectId:"p1",sessionId:"native",sessionKey:"stable",prompt:"task"};
  const assembled=await engine.session_start(ctx); await engine.afterTurn({...ctx,messages:[{role:"assistant",content:"done"}]}); await engine.compact(ctx); await engine.session_end(ctx);
  assert.equal(assembled.packed_context,"ctx"); assert.ok(calls.some(c=>c.url==="/v1/context/assemble")); assert.ok(calls.some(c=>c.url===`/v1/sessions/${stableSessionKey(ctx)}/finalize`)); server.close();
});
test("OpenClaw plugin registers context engine, lifecycle hooks, and explicit tools", () => {
  const tools=[]; const hooks=[]; const engines=[];
  memoryOSPlugin.register({
    pluginConfig:{baseUrl:"http://127.0.0.1:8765"},
    logger:{info:()=>{},warn:()=>{},error:()=>{}},
    registerTool:(tool)=>tools.push(tool.name),
    registerContextEngine:(id)=>engines.push(id),
    on:(name)=>hooks.push(name),
  });
  assert.deepEqual(engines,["memoryos"]);
  assert.deepEqual(new Set(hooks),new Set(["session_start","session_end","before_reset"]));
  assert.deepEqual(new Set(tools),new Set(["memory_recall","memory_store","memory_forget","memory_read","memoryos_status","memoryos_recall_trace"]));
});
