import { readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
const root = process.env.PROTEIN_RESCUE_ROOT || process.cwd()
const versions=['rescue_engineering_v1','pdz2018_information_dev_v1']
const Kind=Symbol.for('TypeBox.Kind')
const string={ [Kind]:'String',type:'string' }
export function scopedVersion(session){
  return versions.find(v=>session.workspaceRoot?.startsWith(root+'/.local/'+v+'/agent_workspaces/'))
}
export function allowTool(payload,session){
  if(!scopedVersion(session))return {allow:true}
  if(['query_epistasis_batch','science_task_state'].includes(payload.name))return {allow:true}
  if(payload.name==='tool_describe' && ['query_epistasis_batch','science_task_state'].includes(payload.args?.name))return {allow:true}
  return {allow:false,reason:'Protein Rescue Lab runtime whitelist: this task permits only scientific query/state tools and their schemas.'}
}
export const scienceRuntime={
  inject:['extension'],
  apply(ctx){
    const api=ctx.extension()
    api.registerHook('tool_call',(payload,context)=>allowTool(payload,context.session))
    async function request(args,context,action){
      const version=scopedVersion(context.session)
      if(!version)throw new Error('Scientific tool unavailable outside dedicated experiment workspaces')
      const endpoint=JSON.parse(await readFile(root+'/.local/'+version+'/oracle_endpoint.json','utf8'))
      const request_id=createHash('sha256').update(context.session.key+'|'+String(context.session.toolUseId)).digest('hex')
      const response=await fetch(endpoint.url,{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({...args,session_key:context.session.key,request_id,action}),signal:context.signal})
      const structured=await response.json()
      return {content:[{type:'text',text:JSON.stringify(structured)}],structured,isError:!response.ok}
    }
    const meta={isReadOnly:false,isDestructive:false,isConcurrencySafe:false,isOpenWorld:false,replay:'never',costHint:undefined,requiresApproval:'never',deferLoading:false}
    api.registerTool({name:'query_epistasis_batch',description:'Commit four distinct unqueried IDs from this task query pool. Predictions are prereveal residual hypotheses in kcal/mol, one per ID. Reveals four historical measurements and charges four of32 queries. Cannot access held evaluation labels. After lost responses, use science_task_state. Stop at done=true.',
      parameters:{ [Kind]:'Object',type:'object',properties:{task_token:{...string,minLength:32,maxLength:32},
        rescue_ids:{ [Kind]:'Array',type:'array',items:string,minItems:4,maxItems:4},
        predicted_residuals:{ [Kind]:'Array',type:'array',items:{[Kind]:'Number',type:'number'},minItems:4,maxItems:4},
        rationale:{...string,maxLength:300}},required:['task_token','rescue_ids','predicted_residuals','rationale'],additionalProperties:false},
      meta,execute:(args,context)=>request(args,context,'query')})
    api.registerTool({name:'science_task_state',description:'Recover current observed scientific state, already-charged query count and legal candidates after errors. No new measurement is revealed or charged. Cannot change or reset the task.',
      parameters:{[Kind]:'Object',type:'object',properties:{task_token:{...string,minLength:32,maxLength:32}},required:['task_token'],additionalProperties:false},
      meta:{...meta,isReadOnly:true,replay:'safe'},execute:(args,context)=>request(args,context,'state')})
  }
}
