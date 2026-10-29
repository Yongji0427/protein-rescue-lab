import { createWriteStream } from 'node:fs'
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import { toolboxClient } from './toolbox_agh_client.mjs'
const root = fileURLToPath(new URL('../', import.meta.url))
const out = resolve(root, 'experiments/science_toolbox_v1')
const privateRoot = resolve(root, '.local/science_toolbox_v1')
const dataset = process.argv[2] ?? 'synthetic'
if (!['synthetic', 'pdz-dev'].includes(dataset)) throw new Error('Use synthetic or pdz-dev')
const protocol = JSON.parse(await readFile(resolve(out, 'protocol.json'), 'utf8'))
const manifest = JSON.parse(await readFile(resolve(out, dataset + '_manifest.json'), 'utf8'))
const taskIds = process.argv[3] ? [process.argv[3]] : manifest.task_ids
if (taskIds.some(id => !manifest.task_ids.includes(id))) throw new Error('Task outside prepared identity cohort')
const client = await toolboxClient()
try {
  const inventory = await client.packages.list({ profile: 'local-dev' })
  if (inventory.packages.find(x => x.id === '@protein-rescue/science-toolbox')?.actual !== 'running') throw new Error('Scientific toolbox not active')
  await mkdir(resolve(out, 'agh_events'), { recursive: true })
  async function run(taskId) {
    const dest = resolve(out, 'agh_events', taskId + '.run.json')
    try { await readFile(dest); console.log(JSON.stringify({ taskId, preserved_existing_trial: true })); return } catch (e) { if (e.code !== 'ENOENT') throw e }
    const input = JSON.parse(await readFile(resolve(privateRoot, 'public_tasks', taskId + '.json'), 'utf8'))
    const endpoint = JSON.parse(await readFile(resolve(privateRoot, 'oracle_endpoint.json'), 'utf8'))
    const sessionKey = 'science_toolbox_v1-' + taskId
    const cwd = resolve(privateRoot, 'agent_workspaces', taskId)
    await mkdir(cwd, { recursive: true }); await client.workspace.add(cwd)
    const session = await client.session.new({ cwd, sessionKey })
    const stream = createWriteStream(resolve(out, 'agh_events', taskId + '.jsonl'), { flags: 'wx' })
    // Keep raw event details private; redact task tokens from exported traces.
    session.listeners.add((method, params) => {
      if (method === '_agnes/v1/session.event') stream.write(JSON.stringify(params.event).replaceAll(input.task_token, '[TASK_TOKEN]') + '\n')
    })
    session.onPermissionRequest(async request => {
      const rejection = request.options.find(x => x.kind.startsWith('reject_'))
      if (!rejection) throw new Error('No rejection option')
      return { optionId: rejection.optionId }
    })
    await session.attach()
    async function state() {
      const response = await fetch(endpoint.url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'science_state', task_token: input.task_token, session_key: sessionKey }) })
      if (!response.ok) throw new Error('Trusted state recovery failed')
      return response.json()
    }
    const base = await readFile(resolve(out, 'prompt.txt'), 'utf8')
    const turns = [], started = Date.now()
    try {
      for (let attempt = 0; attempt <= protocol.max_completion_corrections; attempt++) {
        const before = await state(); if (before.finished) break
        let prompt = attempt === 0 ? base : 'Continue this same scientific task after an operational interruption. State and budgets below are authoritative. Preserve hypotheses and charged queries. Finish the research and use science_finish. No reset or replacement trial.\n' + base
        if (dataset === 'synthetic' && attempt === 0) prompt += '\nEngineering diagnostic: first attempt native todo once with a pending item "toolbox whitelist probe". It should be denied before execution; then proceed with the scientific toolbox and do not repeat it. All numbers and geometry in this task are SYNTHETIC.\n'
        if (taskId === 'synthetic_python_probe') prompt += '\nThis separate diagnostic tests native autonomous programming. Your FIRST scientific tool call must be science_python: write your own analysis of the visible observations and public features, return at least one numeric summary and a proposed candidate ordering. Then record the hypothesis, test a new batch, update the hypothesis, check controls and finish. Do not substitute science_predict for the programming test.\n'
        if (dataset === 'pdz-dev') prompt += '\nEngineering data-use requirement: use science_python at least once, read its data variable rather than retyping observations, and calculate an explicit reproducible summary or candidate scoring function. Never describe manually assigned guesses as calculated additive priors. Record hypotheses before and after new feedback. These are capability checks, not evidence of fully unconstrained behavior.\n'
        let outcome
        const controller = new AbortController()
        const deadline = setTimeout(() => controller.abort(), protocol.turn_timeout_ms)
        let monitorBusy = false, conclusionStop = false
        const monitor = setInterval(async () => {
          if (monitorBusy) return
          monitorBusy = true
          try {
            if ((await state()).finished) { conclusionStop = true; controller.abort() }
          } catch {} finally { monitorBusy = false }
        }, 2000)
        try { outcome = await session.prompt(prompt + '\n' + JSON.stringify(before), { signal: controller.signal }) }
        catch (error) { outcome = { error: String(error.message).replaceAll(input.task_token, '[TASK_TOKEN]').slice(0, 200) } }
        finally { clearTimeout(deadline); clearInterval(monitor) }
        const after = await state(); turns.push({ attempt, before: before.queries_used, after: after.queries_used, finished: after.finished, outcome })
        turns.at(-1).stopped_after_committed_conclusion = conclusionStop
      }
    } finally { await new Promise(done => stream.end(done)) }
    const final = await state(); delete final.task_token
    await mkdir(resolve(out, 'queried_results'), { recursive: true })
    await writeFile(resolve(out, 'queried_results', taskId + '.json'), JSON.stringify(final, null, 2) + '\n')
    const run = { task_id: taskId, dataset, session_id: session.id, queries: final.queries_used,
      python_attempts: final.python_used, finished: final.finished, turns, seconds: (Date.now() - started) / 1000,
      evidence: final.evidence_boundary, replacement_trial: false }
    await writeFile(dest, JSON.stringify(run, null, 2) + '\n')
    console.log(JSON.stringify(run))
  }
  await Promise.all(taskIds.map(run))
} finally { await client.close() }
