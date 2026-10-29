import { readFile,writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { createClient,memoryJournal } from '../vendor/agnes-harness/packages/sdk/src/index.node.ts'
import { inspectPackage,installPreview,waitForPackageOperation,newPackageCommandId } from '../vendor/agnes-harness/packages/cli-tui/src/package-admin.ts'
const root=fileURLToPath(new URL('../',import.meta.url))
const owner=JSON.parse(await readFile('/tmp/protein-rescue-agh-20261008/data/daemon/owner.json','utf8'))
const client=createClient({transport:{kind:'unix',path:owner.socketPath},auth:{kind:'local'},journal:memoryJournal()})
const profile='local-dev'
try{
  await client.initialize()
  let inventory=await client.packages.list({profile})
  let item=inventory.packages.find(x=>x.id==='@protein-rescue/pdz-tools')
  const version=JSON.parse(await readFile(root+'plugins/pdz-tools/package.json','utf8')).version
  if(item && item.version!==version){
    const preview=await inspectPackage(client,profile,{type:'file',ref:'file:./plugins/pdz-tools'})
    const operation=await waitForPackageOperation(client,await client.packages.update({profile,clientId:await client.clientId(),commandId:newPackageCommandId('update'),id:item.id,source:preview.source,expectedIntegrity:preview.integrity}))
    item=operation.installed
  }
  if(!item){
    const preview=await inspectPackage(client,profile,{type:'file',ref:'file:./plugins/pdz-tools'})
    if(preview.blockers.length)throw new Error('PDZ plugin preview blocked')
    item=(await installPreview(client,profile,preview)).installed
  }
  if(!item.trusted)await waitForPackageOperation(client,await client.packages.trust({profile,clientId:await client.clientId(),commandId:newPackageCommandId('trust'),id:item.id,expectedIntegrity:item.integrity,capabilityHash:item.capabilityHash}))
  if(item.actual!=='running')await waitForPackageOperation(client,await client.packages.enable({profile,clientId:await client.clientId(),commandId:newPackageCommandId('enable'),id:item.id}))
  inventory=await client.packages.list({profile})
  const safe=inventory.packages.filter(x=>['@protein-rescue/pdz-tools','@protein-rescue/query-tools'].includes(x.id))
  await writeFile(root+'experiments/pdz2018_point_dev_v1/plugin_inventory.json',JSON.stringify(safe,null,2)+'\n')
  console.log(JSON.stringify(safe.map(x=>({id:x.id,version:x.version,actual:x.actual,trusted:x.trusted,integrity:x.integrity}))))
}finally{await client.close()}
