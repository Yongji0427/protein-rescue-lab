// Read-only replay captures late events after RPC errors; never starts new model work.
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import { toolboxClient } from './toolbox_agh_client.mjs'
const root = fileURLToPath(new URL('../', import.meta.url))
const out = resolve(root, 'experiments/science_toolbox_v1')
const dir = resolve(out, 'final_native_audit')
await mkdir(dir, { recursive: true })
const client = await toolboxClient()
try {
  for (const taskId of ['pdz2__A4L', 'pdz3__H1A', 'syntrophin__H1L']) {
    const privateTask = JSON.parse(await readFile(resolve(root, '.local/science_toolbox_v1/public_tasks', taskId + '.json'), 'utf8'))
    const session = await client.session.load('science_toolbox_v1-' + taskId, {
      cwd: resolve(root, '.local/science_toolbox_v1/agent_workspaces', taskId) })
    const events = new Map()
    session.listeners.add((method, params) => {
      if (method === '_agnes/v1/session.event') events.set(params.event.seq,
        JSON.stringify(params.event).replaceAll(privateTask.task_token, '[TASK_TOKEN]'))
    })
    await session.attach()
    if (taskId === 'pdz2__A4L') await session.cancel() // Stop any late compaction, preserving committed science state.
    await writeFile(resolve(dir, taskId + '.jsonl'), [...events].sort((a,b)=>a[0]-b[0]).map(x=>x[1]).join('\n')+'\n')
    console.log(JSON.stringify({task_id:taskId,replayed_native_events:events.size,new_model_prompts:0}))
  }
} finally { await client.close() }
