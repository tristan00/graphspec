import test from 'node:test';
import assert from 'node:assert/strict';
import {valueOf,parameterFields,displayValue,compareValues,matchesFilter,groupedResults,valueKey} from '../tuning/web/parameters.js';

const trial = (hops,status='complete',mse=1,r2=.2) => ({config:{graph:hops === undefined ? {} : {'derive/hops':hops},model:{enabled:false},epochs:3},status,mse,r2,seconds:12});
const field='config/graph/derive/hops';

test('Every saved parameter is discoverable, including slashes and top-level settings',() => {
  assert.deepEqual(parameterFields([trial(2)]),[field,'config/model/enabled','config/epochs'].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true})));
  assert.equal(valueOf(trial(0),field),0);
  assert.equal(valueOf(trial(2),'config/model/enabled'),false);
});

test('Trial time limits can be found, filtered, sorted and grouped as parameters',() => {
  const rows=[600,1800,1200].map(max_seconds => ({status:'complete',mse:1,config:{max_seconds}}));
  const limit='config/max_seconds';
  assert.deepEqual(parameterFields(rows),[limit]);
  assert.equal(matchesFilter(rows[1],{field:limit,op:'gte',value:'1200'}),true);
  assert.deepEqual(rows.map(row=>valueOf(row,limit)).sort(compareValues),[600,1200,1800]);
  assert.deepEqual(groupedResults(rows,limit,'mse').map(group=>group.value),[600,1200,1800]);
});

test('Numeric parameter sorting preserves numeric order and keeps absent values last',() => {
  assert.deepEqual([10,undefined,0,100,2].sort((a,b)=>compareValues(a,b)),[0,2,10,100,undefined]);
  assert.equal(compareValues(undefined,0,'desc'),1);
  assert.deepEqual([10,0,100,2].sort((a,b)=>compareValues(a,b,'desc')),[100,10,2,0]);
});

test('Missing, null, zero and false are distinct parameter values',() => {
  assert.equal(displayValue(undefined),'Not used');
  assert.equal(displayValue(null),'null');
  assert.equal(displayValue(false),'false');
  assert.equal(displayValue(0),'0');
  assert.equal(new Set([undefined,null,false,0].map(valueKey)).size,4);
});

test('Filters support numeric ranges, boolean values and conditional parameter absence',() => {
  assert.equal(matchesFilter(trial(10),{field,op:'gt',value:'2'}),true);
  assert.equal(matchesFilter(trial(10),{field,op:'lt',value:'2'}),false);
  assert.equal(matchesFilter(trial(0),{field,op:'present',value:''}),true);
  assert.equal(matchesFilter(trial(undefined),{field,op:'missing',value:''}),true);
  assert.equal(matchesFilter(trial(null),{field,op:'missing',value:''}),false);
  assert.equal(matchesFilter(trial(0),{field,op:'gte',value:''}),false);
  assert.equal(matchesFilter(trial(0),{field:'config/model/enabled',op:'eq',value:'false'}),true);
  assert.equal(matchesFilter(trial(undefined),{field,op:'ne',value:'2'}),false);
});

test('Exact group drilldown preserves value types',() => {
  assert.equal(matchesFilter(trial(false),{field,op:'is',value:'false'}),true);
  assert.equal(matchesFilter(trial('false'),{field,op:'is',value:'false'}),false);
});

test('Numeric equality accepts equivalent decimal and exponent input without changing string matching',() => {
  for (const value of ['2','2.0',' 2 ','2e0']) {
    assert.equal(matchesFilter(trial(2),{field,op:'eq',value}),true);
    assert.equal(matchesFilter(trial(2),{field,op:'ne',value}),false);
  }
  for (const value of ['',' ','NaN','Infinity']) assert.equal(matchesFilter(trial(0),{field,op:'eq',value}),false);
  assert.equal(matchesFilter(trial('2'),{field,op:'eq',value:'2.0'}),false);
  assert.equal(matchesFilter(trial(undefined),{field,op:'eq',value:'0'}),false);
});

test('Grouped results count all attempts but score only completed outcomes',() => {
  const rows=[trial(2,'complete',0,.1),trial(2,'complete',4,.7),trial(2,'failed',999,999),trial(undefined,'pending',null,null)];
  const groups=groupedResults(rows,field,'mse');
  assert.equal(groups[0].count,3);
  assert.equal(groups[0].scored,2);
  assert.equal(groups[0].best,0);
  assert.equal(groups[0].median,2);
  assert.equal(groups[1].value,undefined);
  assert.equal(groups[1].best,null);
  assert.equal(groupedResults(rows,field,'r2')[0].best,.7);
  assert.equal(groupedResults(rows,field,'seconds')[0].scored,3);
});

test('User-controlled parameter names cannot read object prototypes',() => {
  const row=trial(2);
  assert.equal(valueOf(row,'config/__proto__'),undefined);
  assert.equal(valueOf(row,'constructor'),undefined);
});

test('Graphs summarize arbitrary numeric targets using their saved values',()=>{
  const rows=[{status:'complete',config:{group:1,target:10},metrics:{gpu:20}},{status:'complete',config:{group:1,target:20},metrics:{gpu:40}}];
  assert.equal(groupedResults(rows,'config/group','config/target')[0].median,15);
  assert.equal(groupedResults(rows,'config/group','result/gpu')[0].median,30);
});
