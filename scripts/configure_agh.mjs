import { readFile, writeFile } from 'node:fs/promises'
import { createClient, memoryJournal } from '../vendor/agnes-harness/packages/sdk/src/index.node.ts'

const root = new URL('../', import.meta.url)
const config = Object.fromEntries((await readFile(new URL('.env', root), 'utf8')).split('\n')
  .filter(line => line.trim() && !line.trim().startsWith('#') && line.includes('='))
  .map(line => { const i = line.indexOf('='); return [line.slice(0, i).trim(), line.slice(i + 1).trim().replace(/^["']|["']$/g, '')] }))
const owner = JSON.parse(await readFile('/tmp/protein-rescue-agh-20261008/data/daemon/owner.json', 'utf8'))
const client = createClient({ transport: { kind: 'unix', path: owner.socketPath }, auth: { kind: 'local' }, journal: memoryJournal() })
try {
  await client.initialize()
  const before = await client.config.get()
  const tested = await client.config.test({ providerId: 'agnes-ai', baseUrl: config.AGNES_BASE_URL, apiKey: config.AGNES_API_KEY })
  if (!tested.models.some(x => x.id === config.AGNES_MODEL)) throw new Error('Configured model not in verified account catalogue')
  const saved = await client.config.save({ providerId: 'agnes-ai', baseUrl: config.AGNES_BASE_URL,
    apiKey: config.AGNES_API_KEY, model: config.AGNES_MODEL, expectedRevision: before.revision })
  const info = { configured: saved.configured, model: config.AGNES_MODEL, provider: 'agnes-ai',
    endpoint: config.AGNES_BASE_URL, accountModels: tested.models.map(x => x.id), revision: saved.revision,
    credentialsLogged: false, home: '/tmp/protein-rescue-agh-20261008' }
  await writeFile(new URL('experiments/agnes_agh_configuration.json', root), JSON.stringify(info, null, 2) + '\n')
  console.log(JSON.stringify(info, null, 2))
} finally { await client.close() }
