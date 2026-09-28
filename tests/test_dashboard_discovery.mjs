import test from 'node:test';
import assert from 'node:assert/strict';
import {hierarchyParts,hierarchyEntries,evaluateParameter,rankParameters} from '../tuning/web/discovery.js';
import {resultFields,valueOf,setMeasures} from '../tuning/web/parameters.js';

setMeasures([{key:'mse',label:'Validation MSE',short:'MSE',better:'lower'},{key:'r2',label:'Validation R²',short:'R²',better:'higher'}]);

const rows = Array.from({length:120},(_,i)=>({id:`trial-${i}`,x:i,y:2+i/10}));

test('Hierarchy supports slash and colon branches while keeping exact field identities',()=>{
  const fields = ['number','mse','config/epochs','config/graph/derive/hops','config/graph/derive/project:client:sources:policy','config/model/hidden'];
  assert.deepEqual(hierarchyParts(fields[4]),['graph','derive','project','client','sources','policy']);
  const root = hierarchyEntries(fields);
  assert.deepEqual(root.folders.map(folder=>folder.name),['graph','model','Results']);
  assert.deepEqual(root.leaves.map(leaf=>leaf.field),['config/epochs']);
  const branch = hierarchyEntries(fields,['graph','derive']);
  assert.deepEqual(branch.folders.map(folder=>folder.name),['project']);
  assert.deepEqual(branch.leaves.map(leaf=>leaf.field),['config/graph/derive/hops']);
  assert.equal(branch.total,2);
});

test('Branch search, multiword paths, result labels and empty matches are exact',()=>{
  const fields = ['mse','number','config/graph/derive/hops','config/model/hops'];
  assert.deepEqual(hierarchyEntries(fields,['graph'],'hops').leaves.map(leaf=>leaf.field),[fields[2]]);
  assert.equal(hierarchyEntries(fields,[],'hops').total,2);
  assert.equal(hierarchyEntries(fields,[],'graph hops').total,1);
  assert.equal(hierarchyEntries(fields,[],'validation mse').leaves[0].field,'mse');
  assert.equal(hierarchyEntries(fields,['graph'],'missing').total,0);
});

test('Held-out prediction detects numeric signal without depending on input order',()=>{
  const result = evaluateParameter(rows);
  assert.ok(result.gain>.65 && result.gain<.85);
  assert.equal(result.improved,5);
  assert.deepEqual(evaluateParameter([...rows].reverse()),result);
  assert.equal(result.used,120);
});

test('Prediction is invariant to metric scale, including tiny and very large values',()=>{
  const result = evaluateParameter(rows);
  for (const scale of [1e-150,1e150,-10]) assert.ok(Math.abs(evaluateParameter(rows.map(row=>({...row,y:row.y*scale}))).gain-result.gain)<1e-10);
});

test('Repeated categories carry signal while unique labels cannot memorize results',()=>{
  const categorical = rows.map((row,i)=>({...row,x:i%2 ? 'a' : 'b',y:i%2 ? 10 : 0}));
  const result = evaluateParameter(categorical);
  assert.ok(result.gain>.97);
  assert.equal(result.type,'Category means');
  assert.ok(evaluateParameter(rows.map(row=>({...row,x:row.id}))).reason);
});

test('A deterministic noise parameter does not appear strongly predictive',()=>{
  let seed=1532;
  const noise=Array.from({length:300},(_,i)=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return {id:`noise-${i}`,x:i,y:seed/4294967296};});
  assert.ok(evaluateParameter(noise).gain<.05);
});

test('Insufficient, constant and zero-valued samples are handled explicitly',()=>{
  assert.ok(evaluateParameter(rows.slice(0,19)).reason);
  assert.equal(evaluateParameter(rows.map(row=>({...row,x:false}))).reason,'Constant recorded value');
  assert.equal(evaluateParameter(rows.map(row=>({...row,y:0}))).reason,'No metric variation');
  assert.equal(evaluateParameter(rows.map(row=>({...row,y:4}))).reason,'No metric variation');
});

test('Ranking groups duplicate fields and excludes absent, null and unmeasured values',()=>{
  const trials = rows.map((row,i)=>({id:row.id,status:'complete',mse:row.y,config:{a:row.x,b:row.x,constant:false,...(i<90 ? {partial:row.x} : {partial:null})}}));
  trials.push({id:'failed',status:'failed',mse:1e8,config:{a:999,b:999,constant:true}});
  const result = rankParameters(trials,'mse');
  assert.equal(result.measured,120);
  const pair = result.rankings.find(row=>row.fields.includes('config/a'));
  assert.deepEqual(pair.fields,['config/a','config/b']);
  assert.equal(pair.used,120);
  assert.equal(result.rankings.find(row=>row.fields.includes('config/partial')).used,90);
  assert.equal(result.skipped['Constant recorded value'],1);
  assert.ok(result.rankings.every(row=>Number.isFinite(row.gain)));
});

test('Duration includes recorded failed attempts, and unsupported criteria fail clearly',()=>{
  const trials=rows.map(row=>({id:row.id,status:'failed',seconds:row.y,config:{x:row.x}}));
  assert.equal(rankParameters(trials,'seconds').measured,120);
  assert.equal(rankParameters(trials,'mse').measured,0);
  assert.throws(()=>rankParameters(trials,'invalid'),/supported metric/);
});

test('Any numeric saved setting or nested result can be a target without ranking itself',()=>{
  const trials=rows.map(row=>({id:row.id,status:'complete',config:{x:row.x,target:row.y,copy:row.y},metrics:{timing:{fit:row.y}}}));
  for (const target of ['config/target','result/timing/fit']) {
    const result=rankParameters(trials,target);
    assert.equal(result.categorical,false);
    assert.equal(result.rankings.length,1);
    assert.deepEqual(result.rankings[0].fields,['config/x']);
    assert.ok(result.rankings[0].gain>.65);
    assert.equal(result.skipped['Target or identical values'],2);
  }
  assert.deepEqual(resultFields(trials),['result/timing/fit']);
  assert.equal(valueOf(trials[0],'result/timing/fit'),2);
  assert.deepEqual(hierarchyParts('result/timing/fit'),['Results','timing','fit']);
});

test('Categorical target prediction is independent of category label order',()=>{
  const trials=rows.map((row,i)=>({id:row.id,status:i<60 ? 'complete' : 'failed',config:{signal:i<60 ? 0 : 1}}));
  const result=rankParameters(trials,'status');
  assert.equal(result.categorical,true);
  assert.ok(result.rankings[0].gain>.99);
  const renamed=trials.map(trial=>({...trial,status:trial.status==='complete' ? 'z' : 'a'}));
  assert.equal(rankParameters(renamed,'status').rankings[0].gain,result.rankings[0].gain);
});

test('Boolean targets work and sparse category targets are explicitly unranked',()=>{
  const trials=rows.map((row,i)=>({id:row.id,config:{signal:i<60 ? 0 : 1,target:i<60,unique:row.id}}));
  const result=rankParameters(trials,'config/target');
  assert.equal(result.categorical,true);
  assert.ok(result.rankings.some(row=>row.fields.includes('config/signal') && row.gain>.99));
  assert.equal(rankParameters(trials,'config/unique').rankings.length,0);
});
