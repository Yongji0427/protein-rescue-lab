// One disclosed operational recovery; retain the original failed run and its budget.
import { createWriteStream } from 'node:fs'
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import { toolboxClient } from './toolbox_agh_client.mjs'
const root = fileURLToPath(new URL('../', import.meta.url))
const out = resolve(root, 'experiments/science_toolbox_v1')
const taskId = process.argv[2]
const phase = process.argv[3] ?? 'operational_recovery'
if (!['operational_recovery', 'controller_completion', 'compacted_completion'].includes(phase)) throw new Error('Unknown engineering recovery phase')
const manifest = JSON.parse(await readFile(resolve(out, 'pdz-dev_manifest.json'), 'utf8'))
if (!manifest.task_ids.includes(taskId)) throw new Error('Recovery task outside prepared cohort')
const original = JSON.parse(await readFile(resolve(out, 'agh_events', taskId + '.run.json'), 'utf8'))
if (original.finished) throw new Error('Completed trials cannot be recovered or replaced')
const folder = resolve(out, phase)
await mkdir(folder, { recursive: true })
const dest = resolve(folder, taskId + '.json')
try { await readFile(dest); throw new Error('Recovery already recorded; do not replace it') } catch (e) { if (e.code !== 'ENOENT') throw e }
const privateRoot = resolve(root, '.local/science_toolbox_v1')
const input = JSON.parse(await readFile(resolve(privateRoot, 'public_tasks', taskId + '.json'), 'utf8'))
const endpoint = JSON.parse(await readFile(resolve(privateRoot, 'oracle_endpoint.json'), 'utf8'))
const sessionKey = 'science_toolbox_v1-' + taskId
async function state() {
  const response = await fetch(endpoint.url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'science_state', task_token: input.task_token, session_key: sessionKey }) })
  if (!response.ok) throw new Error('State recovery failed')
  return response.json()
}
const before = await state()
if (before.finished) throw new Error('Evaluator already completed')
const priorEvents = (await readFile(resolve(out, 'agh_events', taskId + '.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse)
const priorSeq = Math.max(...priorEvents.map(e => e.seq))
const client = await toolboxClient()
const stream = createWriteStream(resolve(folder, taskId + '.jsonl'), { flags: 'wx' })
const controller = new AbortController(), started = Date.now()
let monitorBusy = false, conclusionStop = false
const deadline = setTimeout(() => controller.abort(), 240000)
const monitor = setInterval(async () => {
  if (monitorBusy) return
  monitorBusy = true
  try { if ((await state()).finished) { conclusionStop = true; controller.abort() } } catch {} finally { monitorBusy = false }
}, 2000)
let outcome
try {
  const session = await client.session.load(original.session_id, { cwd: resolve(privateRoot, 'agent_workspaces', taskId) })
  session.onPermissionRequest(async request => ({ optionId: request.options.find(x => x.kind.startsWith('reject_')).optionId }))
  session.listeners.add((method, params) => {
    if (method === '_agnes/v1/session.event' && params.event.seq > priorSeq)
      stream.write(JSON.stringify(params.event).replaceAll(input.task_token, '[TASK_TOKEN]') + '\n')
  })
  await session.attach()
  if (phase === 'compacted_completion') {
    await session.compact('Preserve only the fixed damage/one-rescue constraint, 32-query cap, point0.5 threshold, completed analysis findings, observed IDs, hypotheses and pending conclusion. Do not invent labels. The next user message supplies the authoritative persistent state. Raw failed trials remain retained. Omit repeated full candidate tables and WT coordinate details.')
  }
  const base = await readFile(resolve(out, 'prompt.txt'), 'utf8')
  const instruction = phase !== 'operational_recovery'
    ? 'Engineering completion with the repaired research controller. Continue this SAME session with the SAME measured state and remaining budget. Prior Python analyses and original failed turns are retained. Do not restart the study or repeat lengthy analyses. If best_observed reaches0.5, check its four controls and immediately commit science_finish measured_rescue. Otherwise use existing analyses to select useful remaining queries, then commit budget_exhausted or insufficient_evidence honestly. A plain-text answer is not a committed conclusion. The full legal pool and reference structure remain available through tools on demand.\n'
    : 'Operational recovery after a recorded AGH error. Continue in this SAME session and task. Do not reset budgets, replace trials or discard prior decisions. All previously charged measurements and hypotheses below remain authoritative. Finish the remaining research and submit science_finish. Existing successful Python analyses need not be repeated.\n'
  outcome = await session.prompt(instruction + base + '\n' + JSON.stringify(before), { signal: controller.signal })
} catch (error) {
  outcome = { error: String(error.message).replaceAll(input.task_token, '[TASK_TOKEN]').slice(0, 200) }
} finally {
  clearTimeout(deadline); clearInterval(monitor)
  await new Promise(done => stream.end(done)); await client.close()
}
const after = await state()
const record = { task_id: taskId, original_session_id: original.session_id, before: before.queries_used,
  phase,
  after: after.queries_used, finished: after.finished, outcome, stopped_after_committed_conclusion: conclusionStop,
  seconds: (Date.now() - started) / 1000, original_run_preserved: true, budget_reset: false,
  evidence: 'operational completion addendum; initial failure retained; not an independent model trial' }
await writeFile(dest, JSON.stringify(record, null, 2) + '\n', { flag: 'wx' })
await writeFile(resolve(folder, taskId + '.before.json'), JSON.stringify(original, null, 2) + '\n', { flag: 'wx' })
const exported = { ...after }; delete exported.task_token
await writeFile(resolve(folder, taskId + '.observed.json'), JSON.stringify(exported, null, 2) + '\n', { flag: 'wx' })
console.log(JSON.stringify(record))
