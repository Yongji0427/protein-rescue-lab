import { createWriteStream } from 'node:fs'
import { readFile as read, writeFile as write, mkdir as makeDir } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import { createClient, memoryJournal } from '../vendor/agnes-harness/packages/sdk/src/index.node.ts'

const root = fileURLToPath(new URL('../', import.meta.url))
const out = resolve(root, 'experiments/gb1_agnes_v1')
const protocol = JSON.parse(await read(resolve(out, 'protocol.json'), 'utf8'))
const owner = JSON.parse(await read('/tmp/protein-rescue-agh-20261008/data/daemon/owner.json', 'utf8'))
const client = createClient({ transport: { kind: 'unix', path: owner.socketPath }, auth: { kind: 'local' }, journal: memoryJournal() })
const limit = process.argv.includes('--first-only') ? 1 : protocol.damage_tasks.length
const results = []
let next = 0
await makeDir(resolve(out, 'agh_events'), { recursive: true })
await client.initialize()
try {
  const inventory = await client.packages.list({ profile: 'local-dev' })
  const plugin = inventory.packages.find(x => x.id === '@protein-rescue/query-tools')
  if (plugin?.actual !== 'running') throw new Error('Rescue query plugin must be running before any scientific trial')
  async function run(damage) {
    const input = JSON.parse(await read(resolve(out, 'public_tasks', damage + '.json'), 'utf8'))
    const cwd = resolve(root, '.local/agnes_agent_workspaces', damage)
    await makeDir(cwd, { recursive: true })
    await client.workspace.add(cwd)
    const session = await client.session.new({ cwd, sessionKey: 'gb1-agnes-v1-primary-' + damage })
    const events = createWriteStream(resolve(out, 'agh_events', damage + '.jsonl'))
    session.listeners.add((method, params) => {
      if (method === '_agnes/v1/session.event') events.write(JSON.stringify(params.event) + '\n')
    })
    session.onPermissionRequest(async request => {
      const denied = request.options.find(x => x.kind === 'reject_once' || x.kind === 'reject_always')
      if (!denied) throw new Error('Cannot deny an unrelated tool permission')
      return { optionId: denied.optionId }
    })
    await session.attach()
    const prompt = [
      'You are Protein Rescue Lab, searching the measured GB1 IgG-binding assay by retrospective replay.',
      'A damage mutation must remain. Add exactly one mutation elsewhere. Find an absolute combination binding >=0.5 WT within 32 label queries.',
      'Eight WT-ranked combination labels have already been queried and count toward the budget. Use only the observed labels and public candidate menus below. Do not read files, use shell, call helper tools, search the web, or access other tasks.',
      'Use query_rescue_batch for all six remaining batches. Each call must contain exactly four distinct candidates from the CURRENT menu and its required_WT_anchor. Commit all four before feedback. Choose the other three using contrast evidence, promising absolute binding, and exploration. The fixed_rule_proposal is a suggestion, not a required selection.',
      'Do not stop early after success: continue to budget 32 so all methods are compared fairly. The tool returns the full updated history and next menu. When done=true, stop tool calls and provide a short report from WT, damage, single rescue and queried combination evidence. Do not fabricate labels or mechanisms. Rationale <=100 characters.',
      JSON.stringify(input)
    ].join('\n')
    const start = Date.now()
    let result
    try {
      const outcome = await session.prompt(prompt, { signal: AbortSignal.timeout(180000) })
      result = { damage, session_id: session.id, outcome, seconds: (Date.now()-start)/1000 }
    } catch (error) {
      result = { damage, session_id: session.id, error: String(error.message).slice(0,200), seconds: (Date.now()-start)/1000 }
    }
    events.end()
    await write(resolve(out, 'agh_events', damage + '.run.json'), JSON.stringify(result, null, 2)+'\n')
    results.push(result)
    console.log(JSON.stringify({ damage, completed: results.length, total: limit, ...result }))
  }
  // Sessions are independent; this is concurrency of trials, not multiple scientific agents.
  async function worker() {
    while (next < limit) {
      const damage = protocol.damage_tasks[next++]
      const existing = resolve(out, 'agh_events', damage + '.run.json')
      try { await read(existing); console.log(JSON.stringify({damage, cached_run:true})); continue } catch {}
      await run(damage)
    }
  }
  await Promise.all(Array.from({length: Math.min(4,limit)}, worker))
  await write(resolve(out, 'agh_runs.json'), JSON.stringify(results, null, 2)+'\n')
} finally { await client.close() }
