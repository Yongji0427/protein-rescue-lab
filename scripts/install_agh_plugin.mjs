import { readFile, writeFile } from 'node:fs/promises'
import { createClient, memoryJournal } from '../vendor/agnes-harness/packages/sdk/src/index.node.ts'
const owner = JSON.parse(await readFile('/tmp/protein-rescue-agh-20261008/data/daemon/owner.json', 'utf8'))
const client = createClient({ transport: { kind: 'unix', path: owner.socketPath }, auth: { kind: 'local' }, journal: memoryJournal() })
try {
  await client.initialize()
  const list = await client.packages.list({ profile: 'local-dev' })
  await writeFile(new URL('../experiments/agnes_package_inventory.json', import.meta.url), JSON.stringify(list, null, 2))
  console.log(JSON.stringify(list.packages.filter(x => x.id === '@protein-rescue/query-tools'), null, 2))
} finally { await client.close() }
