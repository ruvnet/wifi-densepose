/** Authored synthetic link-ranking experiment. No hardware actions or probabilities. */
import {createKge, loadKge} from '@ruvector/kge';
import {createHash} from 'node:crypto';
import {readFileSync, writeFileSync, mkdirSync, openSync, closeSync, fstatSync, readSync, chmodSync, constants} from 'node:fs';
import {join, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {performance} from 'node:perf_hooks';
const require = createRequire(import.meta.url);
const root = dirname(fileURLToPath(import.meta.url));
export const SENSORS = Object.freeze(['wifi', 'radar', 'door']);
export const CONFIG = Object.freeze({scorer:'hole', dims:32, seed:42});
export const TRAIN = Object.freeze({epochs:100, lr:0.05, seed:42});
export const hash = bytes => createHash('sha256').update(bytes).digest('hex');
export function fixture() {
  const train = [], heldout = [], contexts = [];
  for (let type=0; type<3; type++) {
    for (let i=0; i<10; i++) {
      const s = `context_${type}_${i}`;
      contexts.push(s);
      train.push({s,r:'observedFeature',o:`feature_${type}`,split:'train'});
      const triple = {s,r:'relevantSensor',o:SENSORS[type],split:'train'};
      if(i<8) train.push(triple); else heldout.push({s,r:triple.r,o:triple.o});
    }
  }
  // Context feature membership is allowed transductive side information;
  // held-out relevantSensor target links never enter either model.
  return {train, heldout, contexts};
}
// Private, constructed once; no caller receives mutable cache handles.
const contextSet = new Set(fixture().contexts);
const expectedGraph = (() => {
  const entities=[], relations=[], triples=[];
  const intern=(items,label)=>{let id=items.indexOf(label);if(id<0){id=items.length;items.push(label);}return id;};
  for(const t of fixture().train) triples.push({s:intern(entities,t.s),r:intern(relations,t.r),o:intern(entities,t.o)});
  return {entities,relations,triples,splits:triples.map(()=> 'train')};
})();
const sourceHashes = () => Object.fromEntries(['ranker.mjs','cli.mjs','package-lock.json'].map(n=>[n,hash(readFileSync(join(root,n)))]));
const same = (a,b) => JSON.stringify(a)===JSON.stringify(b);
export function permutedTraining(triples) {
  const out=triples.map(t=>({...t}));
  const targets=out.filter(t=>t.r==='relevantSensor');
  const labels=targets.map(t=>t.o);
  let state=314159;
  for(let i=labels.length-1;i>0;i--) {
    state=(Math.imul(state,1664525)+1013904223)>>>0;
    const j=state%(i+1); [labels[i],labels[j]]=[labels[j],labels[i]];
  }
  targets.forEach((t,i)=>{t.o=labels[i];});
  return out;
}
export function referenceRank(model, context, k=3) {
  return score(model,context,k,fixture().contexts.includes(context));
}
export function cachedRank(model, context, k=3) {
  return score(model,context,k,contextSet.has(context));
}
// The first frozen paired run failed its p95 gate; preserve the reference default.
export const rank = referenceRank;
function score(model, context, k, admitted) {
  if(!admitted || !Number.isInteger(k) || k<1 || k>3)
    throw new Error('Invalid bounded context or top K');
  const stats=model.stats();
  if(!Number.isInteger(stats.entities) || stats.entities<3 || stats.entities>128)
    throw new Error('Invalid bounded entity population');
  const result=model.predict({s:context,r:'relevantSensor',k:stats.entities,useIndex:false});
  if(result.exact!==true || result.ann!==false) throw new Error('Exhaustive ranking required');
  const candidates=result.candidates.filter(c=>SENSORS.includes(c.entity)).map(c=>({sensorId:c.entity,score:c.score}));
  if(candidates.length!==3 || new Set(candidates.map(c=>c.sensorId)).size!==3 || candidates.some(c=>!Number.isFinite(c.score)))
    throw new Error('Invalid sensor scores');
  candidates.sort((a,b)=>b.score-a.score || SENSORS.indexOf(a.sensorId)-SENSORS.indexOf(b.sensorId));
  return {schemaVersion:1,evidence:'SYNTHETIC',candidates:candidates.slice(0,k)};
}
function evaluate(predict, heldout) {
  const ranks=heldout.map(t=>predict(t.s).findIndex(c=>c.sensorId===t.o)+1);
  if(ranks.some(r=>r<1)) throw new Error('Missing held-out candidate');
  return {queries:ranks.length,mrr:ranks.reduce((a,r)=>a+1/r,0)/ranks.length,
    hitsAt1:ranks.filter(r=>r===1).length/ranks.length,ranks};
}
export function pairedTiming(model, contexts) {
  const rounds=[];
  for(let round=0;round<3;round++) {
    for(let i=0;i<100;i++) {const c=contexts[(i+round)%contexts.length];cachedRank(model,c);referenceRank(model,c);}
    const ms={cached:[],reference:[]};
    for(let i=0;i<1000;i++) {
      const c=contexts[(i+round)%contexts.length];
      const calls=[['cached',cachedRank],['reference',referenceRank]];
      if((i+round)%2) calls.reverse();
      for(const [name,fn] of calls) {const t=performance.now();fn(model,c);ms[name].push(performance.now()-t);}
    }
    const summary={round:round+1,callsPerRuntime:1000,warmupsPerRuntime:100};
    for(const [name,values] of Object.entries(ms)) {
      values.sort((a,b)=>a-b);summary[name]={minMs:values[0],p50Ms:values[499],p95Ms:values[949],
        derivedQueriesPerSecond:1000/(values.reduce((a,b)=>a+b,0)/values.length)};
    }
    summary.bothPercentilesLower=summary.cached.p50Ms<summary.reference.p50Ms && summary.cached.p95Ms<summary.reference.p95Ms;
    rounds.push(summary);
  }
  return {rounds,retainedCachedValidation:false, speedGatePassed:rounds.every(r=>r.bothPercentilesLower),
    decision:'Reference default retained after first frozen paired campaign failed round-one p95; later timings do not promote the candidate',
    gate:'Both p50 and p95 strictly lower in every round; fixed before measurement',
    paired:true,alternatingOrder:true,identicalModelAndInputs:true};
}
function writeNew(path,value) {writeFileSync(path,JSON.stringify(value,null,2)+'\n',{flag:'wx',mode:0o600});}
export function readBounded(path,limit) {
  const fd=openSync(path,constants.O_RDONLY|constants.O_NOFOLLOW|constants.O_NONBLOCK);
  try {const st=fstatSync(fd);if(!st.isFile()||st.size<1||st.size>limit) throw new Error('Expected bounded regular artifact');
    const buffer=Buffer.alloc(limit+1);let size=0;
    while(size<buffer.length) {const n=readSync(fd,buffer,size,buffer.length-size,null);if(!n) break;size+=n;}
    if(size>limit) throw new Error('Oversized artifact');return buffer.subarray(0,size);
  } finally {closeSync(fd);}
}
export async function trainExperiment(output) {
  mkdirSync(output,{recursive:false});
  try {
    const beforeSources=sourceHashes();
    const data=fixture();
    const targetKeys=new Set(data.heldout.map(t=>JSON.stringify([t.s,t.r,t.o])));
    if(data.train.some(t=>targetKeys.has(JSON.stringify([t.s,t.r,t.o])))) throw new Error('Held-out target leakage');
    const models=[];
    const trainingReports=[];
    for(const triples of [data.train,permutedTraining(data.train)]) {
      const model=createKge(CONFIG);
      model.addTriples(triples);
      trainingReports.push(await model.train(TRAIN));
      models.push(model);
    }
    const [model,control]=models;
    const modelBytes=model.save();
    writeFileSync(join(output,'model.json'),modelBytes,{flag:'wx',mode:0o600});
    const counts=Object.fromEntries(SENSORS.map(sensor=>[sensor,data.train.filter(t=>t.r==='relevantSensor'&&t.o===sensor).length]));
    const frequency=SENSORS.map(sensorId=>({sensorId,score:counts[sensorId]})).sort((a,b)=>b.score-a.score || SENSORS.indexOf(a.sensorId)-SENSORS.indexOf(b.sensorId));
    const quality={kge:evaluate(c=>rank(model,c).candidates,data.heldout),
      frequency:evaluate(()=>frequency,data.heldout),permutedGraph:evaluate(c=>rank(control,c).candidates,data.heldout)};
    const packageRoot=dirname(require.resolve('@ruvector/kge/package.json'));
    const pkg=JSON.parse(readFileSync(join(packageRoot,'package.json')));
    if(pkg.version!=='0.2.0') throw new Error('Unexpected package version');
    const nativeNames={'linux-x64':'kge.linux-x64-gnu.node','linux-arm64':'kge.linux-arm64-gnu.node','darwin-x64':'kge.darwin-x64.node','darwin-arm64':'kge.darwin-arm64.node','win32-x64':'kge.win32-x64-msvc.node'};
    const backendArtifact=model.backend==='wasm'?'wasm/ruvector_kge_wasm_bg.wasm':`native/${nativeNames[process.platform+'-'+process.arch]}`;
    const receipt={schemaVersion:1,status:'complete',evidence:'SYNTHETIC',promotionAuthorized:false,
      package:{name:pkg.name,version:pkg.version,engineVersion:model.version,backend:model.backend,backendArtifact,backendArtifactSha256:hash(readFileSync(join(packageRoot,backendArtifact)))},
      config:CONFIG,training:TRAIN,trainingReports,stats:model.stats(),fixtureSha256:hash(JSON.stringify(data)),
      modelSha256:hash(modelBytes),controlModelSha256:hash(control.save()),
      sourceSha256:beforeSources,
      quality,latency:{evidence:'MEASURED',scope:'Full rank including input validation and exact entity scoring; excludes load and training',...pairedTiming(model,data.heldout.map(t=>t.s))},
      environment:{node:process.version,platform:process.platform,arch:process.arch},
      limitations:['Authored synthetic transductive link split; context feature membership is trained, target links are not.',
        'Synthetic feature groups encode the authored task. This is not independent real occupancy or sensor utility evidence.',
        'One frozen configuration and seed; no hyperparameter search, relation composition, model promotion or sensor action.',
        'Scores are uncalibrated ranking scores, not probabilities. Equal-frequency baseline uses fixed wifi/radar/door tie order.']};
    if(!same(beforeSources,sourceHashes())) throw new Error('Sources changed during experiment');
    writeNew(join(output,'receipt.json'),receipt);
    chmodSync(join(output,'receipt.json'),0o400);
    chmodSync(join(output,'model.json'),0o400);
    return {model,receipt};
  } catch(error) {writeNew(join(output,'failed.json'),{status:'failed',promotionAuthorized:false,error:String(error.message)});throw error;}
}
export function loadArtifact(directory) {
  const receipt=JSON.parse(readBounded(join(directory,'receipt.json'),256*1024));
  const bytes=readBounded(join(directory,'model.json'),2*1024*1024);
  if(receipt.status!=='complete'||receipt.evidence!=='SYNTHETIC'||receipt.promotionAuthorized!==false||receipt.package?.version!=='0.2.0'||receipt.modelSha256!==hash(bytes)||receipt.fixtureSha256!==hash(JSON.stringify(fixture())))
    throw new Error('Invalid synthetic artifact provenance');
  const envelope=JSON.parse(bytes);
  const m=envelope.model;
  if(!same(receipt.config,CONFIG)||!same(receipt.training,TRAIN)||!Array.isArray(receipt.trainingReports)||receipt.trainingReports.length!==2
      || receipt.trainingReports.some(t=>t.entities!==36||t.relations!==2||t.triples!==54||t.epochs!==100||t.epoch!==99||!Number.isFinite(t.loss)||!Number.isFinite(t.n3Penalty)))
    throw new Error('Invalid training receipt');
  if(m?.version!==1 || !same(m?.config,CONFIG) || !same(m?.entities?.labels,expectedGraph.entities)
      || !same(m?.relations?.labels,expectedGraph.relations) || !same(m?.triples,expectedGraph.triples)
      || !same(m?.splits,expectedGraph.splits) || m?.tables?.dims!==32
      || !Array.isArray(m?.tables?.entities)||m.tables.entities.length!==1152
      || !Array.isArray(m?.tables?.relations)||m.tables.relations.length!==64
      || [...m.tables.entities,...m.tables.relations].some(x=>!Number.isFinite(x)))
    throw new Error('Invalid authored fixture model envelope');
  return loadKge(bytes.toString());
}
