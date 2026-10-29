import { readFile } from 'node:fs/promises'
const Kind = Symbol.for('TypeBox.Kind')
const string = { [Kind]: 'String', type: 'string' }
export const rescueTools = {
  inject: ['extension'],
  apply(ctx) {
    ctx.extension().registerTool({
      name: 'query_rescue_batch',
      description: 'Commit exactly four unqueried candidates from the current menu, including its required WT anchor. Reveal sealed historical GB1 measurements and charge four label queries. Returns verified feedback and the next menu. Repeat until done=true; then stop. No wet experiment is performed.',
      parameters: { [Kind]: 'Object', type: 'object', properties: {
        task_token: { ...string, minLength: 32, maxLength: 32 },
        rescue_ids: { [Kind]: 'Array', type: 'array', items: string, minItems: 4, maxItems: 4 },
        rationale: { ...string, maxLength: 300 }
      }, required: ['task_token', 'rescue_ids', 'rationale'], additionalProperties: false },
      meta: { isReadOnly: false, isDestructive: false, isConcurrencySafe: false,
        isOpenWorld: false, replay: 'never', costHint: undefined, requiresApproval: 'never', deferLoading: false },
      async execute(args) {
        const endpoint = JSON.parse(await readFile('/tmp/protein-rescue-oracle-20261008.json', 'utf8'))
        const result = await fetch(endpoint.url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(args) })
        const structured = await result.json()
        return { content: [{ type: 'text', text: JSON.stringify(structured) }], structured, isError: !result.ok }
      }
    })
  }
}
