import {test, before, after} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,rmSync,readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const cliPath=fileURLToPath(new URL('./cli.mjs',import.meta.url));
import {fixture,permutedTraining,trainExperiment,loadArtifact,rank,cachedRank,SENSORS,hash} from './ranker.mjs';
let temp, artifact, trained;
before(async()=>{temp=mkdtempSync(join(tmpdir(),'active-kge-test-'));artifact=join(temp,'artifact');trained=await trainExperiment(artifact);});
after(()=>rmSync(temp,{recursive:true,force:true}));
test('held-out target facts absent, known context side facts explicit, control preserves marginals',()=>{
  const f=fixture();assert.equal(f.train.length,54);assert.equal(f.heldout.length,6);
  for(const t of f.heldout) {assert(!f.train.some(x=>x.s===t.s&&x.r===t.r));assert(f.train.some(x=>x.s===t.s&&x.r==='observedFeature'));}
  const p=permutedTraining(f.train);assert.notDeepEqual(p,f.train);
  for(const sensor of SENSORS) assert.equal(p.filter(t=>t.r==='relevantSensor'&&t.o===sensor).length,8);
});
test('actual backend, finite allowlisted ranking, immutable artifact receipt and deterministic seed',async()=>{
  assert(['native','wasm'].includes(trained.model.backend));assert.equal(trained.receipt.package.version,'0.2.0');
  const loaded=loadArtifact(artifact);const a=rank(trained.model,'context_0_8');
  assert.deepEqual(rank(loaded,'context_0_8'),a);
  for(const context of fixture().contexts) for(const k of [1,2,3]) assert.deepEqual(cachedRank(loaded,context,k),rank(loaded,context,k));assert.deepEqual(Object.keys(a),['schemaVersion','evidence','candidates']);
  assert.equal(a.candidates.length,3);for(const c of a.candidates) {assert(SENSORS.includes(c.sensorId));assert(Number.isFinite(c.score));}
  assert.equal(trained.receipt.quality.kge.queries,6);assert.equal(trained.receipt.quality.frequency.hitsAt1,1/3);
  assert.equal(trained.receipt.latency.rounds.length,3);assert(trained.receipt.latency.rounds.every(r=>r.callsPerRuntime===1000));
  assert.equal(trained.receipt.promotionAuthorized,false);
  const second=await trainExperiment(join(temp,'repeat'));
  assert.equal(second.receipt.modelSha256,trained.receipt.modelSha256);
  await assert.rejects(trainExperiment(artifact),/exist/i);
});
test('CLI stdout has only contract JSON; bad input rejected',()=>{
  const run=spawnSync(process.execPath,[cliPath,'--artifact',artifact,'--context','context_1_8','--k','2'],{encoding:'utf8'});
  assert.equal(run.status,0,run.stderr);assert.equal(run.stderr,'');const out=JSON.parse(run.stdout);assert.equal(out.candidates.length,2);
  for(const [context,k] of [['unknown',3],['context_0_8',0],['context_0_8',4],['context_0_8',NaN]]) assert.throws(()=>rank(trained.model,context,k),/Invalid/);
  const bad=spawnSync(process.execPath,[cliPath,'--artifact',artifact,'--k','100000'],{encoding:'utf8'});assert.equal(bad.status,1);assert.equal(bad.stdout,'');
});
test('corrupt artifact rejected before backend load',()=>{
  const dir=join(temp,'corrupt');mkdirSync(dir);writeFileSync(join(dir,'receipt.json'),readFileSync(join(artifact,'receipt.json')));
  writeFileSync(join(dir,'model.json'),readFileSync(join(artifact,'model.json'))+' ');
  assert.throws(()=>loadArtifact(dir),/provenance/);
});

test('rehashing cannot admit altered graph vocabulary, indices, splits or missing training report',()=>{
  const mutations=[
    m=>{m.triples[0].s=999;},
    m=>{m.entities.labels[0]='untrusted_context';},
    m=>{m.relations.labels[0]='otherRelation';},
    m=>{m.splits[0]='test';},
    m=>{delete m.splits;},
    m=>{m.config.seed=43;},
    m=>{const held=fixture().heldout[0];m.triples[0]={s:m.entities.labels.indexOf(held.s),r:m.relations.labels.indexOf(held.r),o:m.entities.labels.indexOf(held.o)};},
  ];
  mutations.forEach((mutate,i)=>{
    const dir=join(temp,'altered-'+i);mkdirSync(dir);
    const envelope=JSON.parse(readFileSync(join(artifact,'model.json')));mutate(envelope.model);
    const bytes=JSON.stringify(envelope),receipt=JSON.parse(readFileSync(join(artifact,'receipt.json')));receipt.modelSha256=hash(bytes);
    writeFileSync(join(dir,'model.json'),bytes);writeFileSync(join(dir,'receipt.json'),JSON.stringify(receipt));
    assert.throws(()=>loadArtifact(dir),/authored fixture/);
  });
  const dir=join(temp,'missing-training');mkdirSync(dir);
  const receipt=JSON.parse(readFileSync(join(artifact,'receipt.json')));delete receipt.trainingReports;
  writeFileSync(join(dir,'model.json'),readFileSync(join(artifact,'model.json')));
  writeFileSync(join(dir,'receipt.json'),JSON.stringify(receipt));
  assert.throws(()=>loadArtifact(dir),/training receipt/);
});
