// Uses an isolated copy of the official SDK; frozen vendor sources are never repaired in place.
import { readFile } from 'node:fs/promises'
import { createClient, memoryJournal } from '../.local/science_toolbox_v1/agh-runtime/packages/sdk/src/index.node.ts'
export { inspectPackage, installPreview, waitForPackageOperation, newPackageCommandId } from '../.local/science_toolbox_v1/agh-runtime/packages/cli-tui/src/package-admin.ts'
export async function toolboxClient() {
  const owner = JSON.parse(await readFile('/tmp/protein-rescue-agh-20261008/data/daemon/owner.json', 'utf8'))
  const client = createClient({ transport: { kind: 'unix', path: owner.socketPath }, auth: { kind: 'local' }, journal: memoryJournal() })
  await client.initialize()
  return client
}
