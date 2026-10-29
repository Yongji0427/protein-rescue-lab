import { readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { resolve } from 'node:path'

const root = process.env.PROTEIN_RESCUE_ROOT || process.cwd()
const version = 'science_toolbox_v1'
const Kind = Symbol.for('TypeBox.Kind')
const str = { [Kind]: 'String', type: 'string' }
const array = (items, maxItems) => ({ [Kind]: 'Array', type: 'array', items, maxItems })
const number = { [Kind]: 'Number', type: 'number' }
const tools = [
  ['science_state', 'Recover task, observed measurements, budgets, hypotheses and conclusion. Free; no new labels.', {}],
  ['science_candidates', 'Enumerate the full measured legal candidate pool, filter by site or amino-acid group, rank public single scores or WT distances. Coverage is public; unqueried outcomes are hidden.', {
    site: { [Kind]: 'Integer', type: 'integer', minimum: 1, maximum: 9 },
    group: { ...str, enum: ['hydrophobic', 'aromatic', 'polar', 'basic', 'acidic', 'special'] },
    rank: { ...str, enum: ['wt', 'distance', 'id'] }
  }],
  ['science_sequence', 'Inspect the scanned sequence and public single-mutant features. Additive scores are predictions, not double-mutant measurements.', {}],
  ['science_structure', 'Inspect validated WT PDB residue mapping and side-chain contact distances. These describe reference geometry, not mutant stability or causal rescue.', {}],
  ['science_predict', 'Rank unqueried candidates with an additive prior or ridge residual fit ONLY to this task observed labels. Leave-one-out error is diagnostic; uncertainty is not a biological confidence interval.', {
    method: { ...str, enum: ['additive', 'ridge'] }
  }],
  ['science_python', 'Write Python using data (public candidates, structure, observed labels and hypotheses), np and pd. Set JSON-serializable result and optionally print. NumPy/pandas/SciPy available. OS denies user files, writes, network and subprocesses. 10s and 12KB code limits; every attempt costs one analysis credit.', {
    code: { ...str, minLength: 1, maxLength: 12000 }, purpose: { ...str, minLength: 1, maxLength: 500 }
  }],
  ['science_record', 'Append a hypothesis or update one with references to observed candidate IDs. State proposed/supported/contradicted/inconclusive. Records are interpretations; structural claims remain hypotheses.', {
    hypothesis_id: { ...str, maxLength: 64 }, claim: { ...str, minLength: 1, maxLength: 1000 },
    status: { ...str, enum: ['proposed', 'supported', 'contradicted', 'inconclusive'] },
    evidence_ids: array(str, 32), next_test: { ...str, maxLength: 1000 }
  }],
  ['science_query', 'Commit 1..4 unique LEGAL unqueried rescue IDs, retaining the damage at its original site. Predictions and rationale are recorded BEFORE returning actual historical measurements. Charge only new labels; whole accepted batch is idempotent. Stop when evidence is sufficient or budget exhausted.', {
    rescue_ids: { ...array(str, 4), minItems: 1 },
    predicted_affinities: { ...array({ ...number, minimum: 0 }, 4), minItems: 1 },
    rationale: { ...str, minLength: 1, maxLength: 1000 }
  }],
  ['science_controls', 'Compare WT, damage, rescue single and already measured damage+rescue. Check legal edits and point recovery. Non-additivity and structure proximity do not prove mechanism.', {
    rescue_id: str
  }],
  ['science_finish', 'Persist final conclusion: measured rescue, budget exhausted, or insufficient evidence. A success must identify a queried candidate meeting the fixed point threshold. Locks further research mutations.', {
    outcome: { ...str, enum: ['measured_rescue', 'budget_exhausted', 'insufficient_evidence'] },
    rescue_id: str, summary: { ...str, minLength: 1, maxLength: 2000 },
    unresolved: { ...str, maxLength: 2000 }
  }]
]
export const toolNames = tools.map(([name]) => name)
export function scoped(session) {
  const folder = resolve(root, '.local', version, 'agent_workspaces') + '/'
  return typeof session.workspaceRoot === 'string' && resolve(session.workspaceRoot).startsWith(folder)
}
export function allowTool(payload, session) {
  if (!scoped(session)) return { allow: true }
  if (toolNames.includes(payload.name)) return { allow: true }
  if (payload.name === 'tool_describe' && toolNames.includes(payload.args?.name)) return { allow: true }
  return { allow: false, reason: 'Scientific toolbox task: use registered science tools; Python analysis runs through science_python.' }
}
export function researchContext(state) {
  const best = state.best_observed
  return 'Protein Rescue Lab research controller: use only ' + toolNames.join(', ') +
    '. Tool results are authoritative; predictions are not measurements. ' +
    'Remaining historical queries: ' + state.queries_remaining + '. Python attempts: ' + state.python_used + '/12. ' +
    (best ? 'Best ALREADY MEASURED combination: ' + best.id + ' affinity=' + best.affinity + '. ' : '') +
    (state.measured_point_recovery_found ? 'A qualifying point rescue already exists. Check science_controls for that observed candidate and submit science_finish with measured_rescue. Do not overlook shared initial observations. Mechanism may remain unresolved. '
      : 'Choose the next useful research step; when budget is exhausted or evidence insufficient, submit science_finish honestly. ') +
    'A plain-text final answer does not commit the research conclusion. Use science_finish before ending.'
}
export function stoppingDecision(state, reason) {
  if (state.finished || reason !== 'completed') return { action: 'stop' }
  return { action: 'continue', note: 'Research conclusion not committed. ' + researchContext(state) }
}
async function currentState(context) {
  const taskId = context.session.key.slice((version + '-').length)
  if (!context.session.key.startsWith(version + '-') || !/^[A-Za-z0-9_]+$/.test(taskId)) throw new Error('Invalid scientific session identity')
  const task = JSON.parse(await readFile(resolve(root, '.local', version, 'public_tasks', taskId + '.json'), 'utf8'))
  const endpoint = JSON.parse(await readFile(resolve(root, '.local', version, 'oracle_endpoint.json'), 'utf8'))
  const response = await fetch(endpoint.url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'science_state', task_token: task.task_token, session_key: context.session.key }),
    signal: AbortSignal.timeout(1500) })
  if (!response.ok) throw new Error('Scientific state unavailable')
  return response.json()
}
export const scienceToolbox = {
  inject: ['extension'],
  apply(ctx) {
    const api = ctx.extension()
    api.registerHook('tool_call', (payload, context) => allowTool(payload, context.session))
    api.registerHook('context', async (_payload, context) => {
      if (!scoped(context.session)) return {}
      try { return { additionalContext: researchContext(await currentState(context)) } }
      catch { return { additionalContext: 'Scientific state temporarily unavailable. Recover with science_state; do not invent completion.' } }
    })
    api.registerHook('turn_stopping', async (payload, context) => {
      if (!scoped(context.session)) return
      try { return stoppingDecision(await currentState(context), payload.proposedReason) }
      catch { return payload.proposedReason === 'completed' ? { action: 'continue', note: 'Recover with science_state and commit science_finish before ending.' } : { action: 'stop' } }
    })
    for (const [name, description, properties] of tools) {
      const required = name === 'science_python' ? ['code', 'purpose']
        : name === 'science_record' ? ['claim', 'status', 'evidence_ids', 'next_test']
        : name === 'science_query' ? ['rescue_ids', 'predicted_affinities', 'rationale']
        : name === 'science_controls' ? ['rescue_id']
        : name === 'science_finish' ? ['outcome', 'summary', 'unresolved'] : []
      api.registerTool({ name, description,
        parameters: { [Kind]: 'Object', type: 'object', properties: {
          task_token: { ...str, minLength: 32, maxLength: 32 }, ...properties
        }, required: ['task_token', ...required], additionalProperties: false },
        meta: { isReadOnly: ['science_state', 'science_candidates', 'science_sequence', 'science_structure', 'science_controls'].includes(name),
          isDestructive: false, isConcurrencySafe: false, isOpenWorld: false, replay: 'never',
          costHint: undefined, requiresApproval: 'never', deferLoading: false },
        async execute(args, context) {
          if (!scoped(context.session)) throw new Error('Toolbox unavailable outside its dedicated workspace')
          const endpoint = JSON.parse(await readFile(resolve(root, '.local', version, 'oracle_endpoint.json'), 'utf8'))
          const request_id = createHash('sha256').update(context.session.key + '|' + String(context.session.toolUseId)).digest('hex')
          const response = await fetch(endpoint.url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...args, action: name, session_key: context.session.key, request_id }), signal: context.signal })
          const structured = await response.json()
          return { content: [{ type: 'text', text: JSON.stringify(structured) }], structured, isError: !response.ok }
        }
      })
    }
  }
}
