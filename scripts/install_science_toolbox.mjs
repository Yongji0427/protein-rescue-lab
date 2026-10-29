import { mkdir, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { toolboxClient, inspectPackage, installPreview, waitForPackageOperation, newPackageCommandId } from './toolbox_agh_client.mjs'
const root = fileURLToPath(new URL('../', import.meta.url))
const client = await toolboxClient()
const profile = 'local-dev', id = '@protein-rescue/science-toolbox'
try {
  let inventory = await client.packages.list({ profile })
  let item = inventory.packages.find(x => x.id === id)
  const preview = await inspectPackage(client, profile, { type: 'file', ref: 'file:./plugins/science-toolbox' })
  if (preview.blockers.length) throw new Error('Toolbox plugin inspection blocked')
  if (item && item.integrity !== preview.integrity) {
    item = (await waitForPackageOperation(client, await client.packages.update({ profile,
      clientId: await client.clientId(), commandId: newPackageCommandId('update'), id,
      source: preview.source, expectedIntegrity: preview.integrity }))).installed
  }
  if (!item) item = (await installPreview(client, profile, preview)).installed
  if (!item.trusted) await waitForPackageOperation(client, await client.packages.trust({ profile,
    clientId: await client.clientId(), commandId: newPackageCommandId('trust'), id,
    expectedIntegrity: item.integrity, capabilityHash: item.capabilityHash }))
  if (item.actual !== 'running') await waitForPackageOperation(client, await client.packages.enable({ profile,
    clientId: await client.clientId(), commandId: newPackageCommandId('enable'), id }))
  inventory = await client.packages.list({ profile })
  const installed = inventory.packages.find(x => x.id === id)
  if (installed?.actual !== 'running') throw new Error('Toolbox not active')
  const safe = { id, version: installed.version, actual: installed.actual, trusted: installed.trusted,
    integrity: installed.integrity, capabilityHash: installed.capabilityHash }
  await mkdir(root + 'experiments/science_toolbox_v1', { recursive: true })
  await writeFile(root + 'experiments/science_toolbox_v1/plugin_inventory.json', JSON.stringify(safe, null, 2) + '\n')
  console.log(JSON.stringify(safe))
} finally { await client.close() }
