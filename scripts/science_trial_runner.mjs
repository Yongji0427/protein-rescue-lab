import { createWriteStream } from 'node:fs'
import { readFile,writeFile,mkdir } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import { createClient,memoryJournal } from '../vendor/agnes-harness/packages/sdk/src/index.node.ts'
const root=fileURLToPath(new URL('../',import.meta.url))
const out=resolve(root,process.argv[2]);const p=JSON.parse(await readFile(resolve(out,'protocol.json'),'utf8'))
const promptText=await readFile(resolve(out,'prompt.txt'),'utf8')
const owner=JSON.parse(await readFile('/tmp/protein-rescue-agh-20261008/data/daemon/owner.json','utf8'))
const client=createClient({transport:{kind:'unix',path:owner.socketPath},auth:{kind:'local'},journal:memoryJournal()})
await client.initialize();let next=0
try{
  const inventory=await client.packages.list({profile:'local-dev'})
  if(inventory.packages.find(x=>x.id==='@protein-rescue/science-runtime')?.actual!=='running')throw new Error('Runtime guard unavailable')
  await mkdir(resolve(out,'agh_events'),{recursive:true})
  async function run(taskId){
    const dest=resolve(out,'agh_events',taskId+'.run.json')
    try{await readFile(dest);console.log(JSON.stringify({taskId,cached:true}));return}catch{}
    const input=JSON.parse(await readFile(resolve(out,'public_tasks',taskId+'.json'),'utf8'))
    const cwd=resolve(root,'.local',p.version,'agent_workspaces',taskId)
    await mkdir(cwd,{recursive:true});await client.workspace.add(cwd)
    const session=await client.session.new({cwd,sessionKey:p.version+'-'+taskId})
    const stream=createWriteStream(resolve(out,'agh_events',taskId+'.jsonl'))
    session.listeners.add((method,params)=>{if(method==='_agnes/v1/session.event')stream.write(JSON.stringify(params.event)+'\n')})
    session.onPermissionRequest(async r=>{const reject=r.options.find(x=>x.kind.startsWith('reject_'));if(!reject)throw new Error('No refusal option');return {optionId:reject.optionId}})
    await session.attach();const turns=[];const start=Date.now()
    async function current(){
      const endpoint=JSON.parse(await readFile(resolve(root,'.local',p.version,'oracle_endpoint.json'),'utf8'))
      const response=await fetch(endpoint.url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'state',task_token:input.task_token,session_key:p.version+'-'+taskId})})
      if(!response.ok)throw new Error('Cannot recover trusted budget state')
      return await response.json()
    }
    for(let attempt=0;attempt<=p.max_completion_corrections;attempt++){
      const state=await current();if(state.done)break
      const text=attempt===0?promptText:
        'Continue in this SAME task after an operational interruption. The state below is authoritative and includes all already charged labels. Do not reset, repeat completed batches, use other tools or fabricate completion. Complete the remaining real query_epistasis_batch calls and prereveal predictions. No fallback or replacement trials.'
      let outcome
      try{outcome=await session.prompt(text+'\n'+JSON.stringify(state),{signal:AbortSignal.timeout(p.turn_timeout_ms)})}
      catch(error){outcome={error:String(error.message).slice(0,160)}}
      const after=await current();turns.push({attempt,before:state.queries_used,after:after.queries_used,outcome})
    }
    const state=await current();await new Promise(done=>stream.end(done))
    const result={task_id:taskId,session_id:session.id,queries:state.queries_used,complete:state.done,turns,seconds:(Date.now()-start)/1000,fallback_used:false}
    await writeFile(dest,JSON.stringify(result,null,2)+'\n')
    console.log(JSON.stringify({taskId,queries:state.queries_used,turns:turns.length,seconds:result.seconds}))
  }
  async function worker(){while(next<p.task_ids.length)await run(p.task_ids[next++])}
  await Promise.all(Array.from({length:Math.min(4,p.task_ids.length)},worker))
}finally{await client.close()}
