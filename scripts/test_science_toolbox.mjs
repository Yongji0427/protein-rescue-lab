import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { allowTool, toolNames, stoppingDecision, researchContext, scienceToolbox } from '../plugins/science-toolbox/index.mjs'
const root = fileURLToPath(new URL('../', import.meta.url))
const session = { workspaceRoot: root + '.local/science_toolbox_v1/agent_workspaces/probe' }
for (const name of toolNames) assert.equal(allowTool({ name }, session).allow, true)
for (const name of ['shell', 'todo', 'query_rescue_batch', 'science_task_state']) assert.equal(allowTool({ name }, session).allow, false)
assert.equal(allowTool({ name: 'tool_describe', args: { name: 'science_python' } }, session).allow, true)
assert.equal(allowTool({ name: 'tool_describe', args: { name: 'shell' } }, session).allow, false)
assert.equal(allowTool({ name: 'todo' }, { workspaceRoot: root }).allow, true)
const state = { finished: false, queries_remaining: 24, python_used: 2,
  best_observed: { id: 'E2A', affinity: .8 }, measured_point_recovery_found: true }
assert.equal(stoppingDecision(state, 'completed').action, 'continue')
assert.equal(stoppingDecision({ ...state, finished: true }, 'completed').action, 'stop')
assert.equal(stoppingDecision(state, 'budget').action, 'stop')
assert.match(researchContext(state), /E2A/); assert.match(researchContext(state), /science_finish/)
const hooks = new Map(), definitions = []
scienceToolbox.apply({ extension: () => ({ registerHook: (name, handler) => hooks.set(name, handler),
  registerTool: definition => definitions.push(definition) }) })
assert.equal(definitions.length, 10)
assert.deepEqual([...hooks.keys()], ['tool_call', 'context', 'turn_stopping'])
assert.equal(await hooks.get('turn_stopping')({ proposedReason: 'completed' }, { session: { workspaceRoot: root } }), undefined)
for (const tool of definitions) { assert.ok(tool.parameters.required.includes('task_token')); assert.ok(Object.hasOwn(tool.meta, 'costHint')) }
const result = { status: 'passed', registered_tools: 10, runtime_scope_guard: true,
  conclusion_controller: true, other_workspace_stopping_unchanged: true, tool_schemas: true }
await mkdir(root + 'experiments/science_toolbox_v1', { recursive: true })
await writeFile(root + 'experiments/science_toolbox_v1/plugin_verification.json', JSON.stringify(result, null, 2) + '\n')
console.log(JSON.stringify(result))
