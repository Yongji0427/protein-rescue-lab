import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { createWriteStream } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import { createClient, memoryJournal } from '../vendor/agnes-harness/packages/sdk/src/index.node.ts'

const root = fileURLToPath(new URL('../', import.meta.url))
const out = resolve(root, 'experiments/gb1_agnes_v1')
const protocol = JSON.parse(await readFile(resolve(out, 'protocol.json'), 'utf8'))
const owner = JSON.parse(await readFile('/tmp/protein-rescue-agh-20261008/data/daemon/owner.json', 'utf8'))
const client = createClient({ transport: { kind: 'unix', path: owner.socketPath }, auth: { kind: 'local' }, journal: memoryJournal() })
await client.initialize()
const corrections = []
await mkdir(resolve(out, 'first_turn_results'), { recursive: true })
try {
  for (const damage of protocol.damage_tasks) {
    const statePath = resolve(root, '.local/agnes_evaluator', damage + '.json')
    const original = await readFile(statePath, 'utf8')
    await writeFile(resolve(out, 'first_turn_results', damage + '.json'), original)
    let state = JSON.parse(original)
    if (state.queries.length === 32) continue
    const trial = JSON.parse(await readFile(resolve(out, 'agh_events', damage + '.run.json'), 'utf8'))
    const cwd = resolve(root, '.local/agnes_agent_workspaces', damage)
    const session = await client.session.load(trial.session_id, { cwd, onPermissionRequest: async request => {
      const option = request.options.find(x => x.kind.startsWith('reject_'))
      if (!option) throw new Error('No reject permission option')
      return { optionId: option.optionId }
    } })
    const stream = createWriteStream(resolve(out, 'agh_events', damage + '.jsonl'), { flags: 'a' })
    session.listeners.add((method, params) => { if (method === '_agnes/v1/session.event') stream.write(JSON.stringify(params.event)+'\n') })
    await session.attach()
    for (let attempt=1; attempt<=2 && state.queries.length<32; attempt++) {
      const before = state.queries.length
      const prompt = `Budget verifier: only ${before}/32 labels were actually queried. Plain-text JSON/XML that resembles a tool call does not execute and reveals no label. Use the real query_rescue_batch function/tool interface provided by AGH. Continue in this SAME task with the current tool-returned menu and required WT anchor until queries_used=32 and done=true. Do not claim completion before the evaluator confirms it. Use no files, shell, or other tools. The fixed damage, objective, candidate rules and query budget are unchanged.`
      const start = Date.now()
      let outcome
      try { outcome = await session.prompt(prompt, { signal: AbortSignal.timeout(180000) }) }
      catch(error) { outcome = { error: String(error.message).slice(0,160) } }
      state = JSON.parse(await readFile(statePath, 'utf8'))
      corrections.push({ damage, attempt, before, after: state.queries.length, outcome, seconds: (Date.now()-start)/1000 })
      console.log(JSON.stringify(corrections.at(-1)))
    }
    stream.end()
  }
  await writeFile(resolve(out, 'completion_corrections.json'), JSON.stringify(corrections,null,2)+'\n')
} finally { await client.close() }
