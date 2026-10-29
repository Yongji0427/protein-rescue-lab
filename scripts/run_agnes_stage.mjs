import { createWriteStream } from 'node:fs'
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import { createClient, memoryJournal } from '../vendor/agnes-harness/packages/sdk/src/index.node.ts'

const root = fileURLToPath(new URL('../',import.meta.url))
const out = resolve(root,process.argv[2])
const p = JSON.parse(await readFile(resolve(out,'protocol.json'),'utf8'))
const promptText = await readFile(resolve(out,'prompt.txt'),'utf8')
const owner = JSON.parse(await readFile('/tmp/protein-rescue-agh-20261008/data/daemon/owner.json','utf8'))
const client = createClient({transport:{kind:'unix',path:owner.socketPath},auth:{kind:'local'},journal:memoryJournal()})
let next=0, finished=0
await mkdir(resolve(out,'agh_events'),{recursive:true})
await mkdir(resolve(out,'first_turn_results'),{recursive:true})
await client.initialize()
try {
  const inventory = await client.packages.list({profile:'local-dev'})
  if(inventory.packages.find(x=>x.id==='@protein-rescue/query-tools')?.actual!=='running') throw new Error('Query plugin not running')
  async function run(damage) {
    const resultPath=resolve(out,'agh_events',damage+'.run.json')
    try {await readFile(resultPath);console.log(JSON.stringify({damage,cached:true}));return}catch{}
    const input=JSON.parse(await readFile(resolve(out,'public_tasks',damage+'.json'),'utf8'))
    const cwd=resolve(root,'.local',p.version,'agent_workspaces',damage)
    await mkdir(cwd,{recursive:true});await client.workspace.add(cwd)
    const session=await client.session.new({cwd,sessionKey:p.version+'-'+damage})
    const stream=createWriteStream(resolve(out,'agh_events',damage+'.jsonl'))
    session.listeners.add((method,params)=>{if(method==='_agnes/v1/session.event')stream.write(JSON.stringify(params.event)+'\n')})
    session.onPermissionRequest(async request=>{
      const option=request.options.find(x=>x.kind.startsWith('reject_'))
      if(!option)throw new Error('Cannot reject unrelated tool')
      return {optionId:option.optionId}
    })
    await session.attach()
    const statePath=resolve(root,'.local',p.version,'evaluator',damage+'.json')
    const start=Date.now();const turns=[]
    for(let attempt=0;attempt<=p.max_completion_corrections;attempt++) {
      const before=JSON.parse(await readFile(statePath,'utf8')).queries.length
      if(before===32)break
      const prompt=attempt===0?promptText+'\n'+JSON.stringify(input):
        `Budget verifier: only ${before}/32 labels were actually queried. Text that resembles a call does not execute. Continue in the SAME task with the real query_rescue_batch tool and the current menu/required WT anchor until queries_used=32 and done=true. No files, shell, web, other tasks or tools. Rules and objective are unchanged. Do not fabricate completion.`
      let outcome
      try {outcome=await session.prompt(prompt,{signal:AbortSignal.timeout(p.turn_timeout_ms)})}
      catch(error){outcome={error:String(error.message).slice(0,160)}}
      const state=JSON.parse(await readFile(statePath,'utf8'))
      turns.push({attempt,before,after:state.queries.length,outcome})
      if(attempt===0)await writeFile(resolve(out,'first_turn_results',damage+'.json'),JSON.stringify(state,null,2)+'\n')
    }
    const state=JSON.parse(await readFile(statePath,'utf8'))
    const result={damage,session_id:session.id,queries:state.queries.length,complete:state.queries.length===32,turns,seconds:(Date.now()-start)/1000}
    await new Promise(resolveEnd=>stream.end(resolveEnd))
    await writeFile(resultPath,JSON.stringify(result,null,2)+'\n')
    console.log(JSON.stringify({damage,finished:++finished,total:p.damage_tasks.length,queries:result.queries,turns:turns.length,seconds:result.seconds}))
  }
  async function worker(){while(next<p.damage_tasks.length)await run(p.damage_tasks[next++])}
  await Promise.all(Array.from({length:Math.min(4,p.damage_tasks.length)},worker))
}finally{await client.close()}
