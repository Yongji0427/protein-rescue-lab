import { readFile } from 'node:fs/promises'
const Kind=Symbol.for('TypeBox.Kind')
const string={ [Kind]:'String',type:'string' }
export const pdzTools={
  inject:['extension'],
  apply(ctx){
    ctx.extension().registerTool({
      name:'query_pdz_batch',
      description:'Commit exactly four new candidates from the current PDZ task menu including required_WT_anchor. Charge four historical label queries and return actual experimental binding energy, point relative affinity, optional count sensitivity and next menu. Complete 32 total; no wet experiment.',
      parameters:{ [Kind]:'Object',type:'object',properties:{
        task_token:{...string,minLength:32,maxLength:32},
        rescue_ids:{ [Kind]:'Array',type:'array',items:string,minItems:4,maxItems:4},
        rationale:{...string,maxLength:300}
      },required:['task_token','rescue_ids','rationale'],additionalProperties:false},
      meta:{isReadOnly:false,isDestructive:false,isConcurrencySafe:false,isOpenWorld:false,replay:'never',costHint:undefined,requiresApproval:'never',deferLoading:false},
      async execute(args){
        const endpoint=JSON.parse(await readFile((process.env.PROTEIN_RESCUE_ROOT || process.cwd()) + '/.local/pdz2018_point_dev_v1/oracle_endpoint.json','utf8'))
        const response=await fetch(endpoint.url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(args)})
        const structured=await response.json()
        return {content:[{type:'text',text:JSON.stringify(structured)}],structured,isError:!response.ok}
      }
    })
  }
}
