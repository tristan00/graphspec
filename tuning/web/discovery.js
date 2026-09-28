import {parameterFields,valueOf,valueKey,RESULTS} from './parameters.js';

export function hierarchyParts(field) {
  return field.startsWith('config/') ? field.slice(7).split(/[/:]/) : ['Results',...(field.startsWith('result/') ? field.slice(7).split(/[/:]/) : [field])];
}

export function hierarchyEntries(fields,path=[],query='') {
  const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
  const matches = fields.filter(field => {
    const parts = hierarchyParts(field);
    return path.every((part,index) => parts[index] === part) && terms.every(term => field.toLowerCase().includes(term) || (RESULTS[field] || '').toLowerCase().includes(term));
  });
  const folders = new Map(), leaves = [];
  for (const field of matches) {
    const parts = hierarchyParts(field);
    if (query.trim() || parts.length <= path.length+1) leaves.push({field,name:parts.at(-1)});
    else {
      const name = parts[path.length];
      if (!folders.has(name)) folders.set(name,{name,path:[...path,name],count:0});
      folders.get(name).count++;
    }
  }
  return {folders:[...folders.values()].sort((a,b)=>a.name.localeCompare(b.name)),leaves,total:matches.length};
}

function hash(text) {
  let value = 2166136261;
  for (const char of text) value = Math.imul(value ^ char.charCodeAt(0),16777619);
  value ^= value >>> 16;
  value = Math.imul(value,0x85ebca6b);
  value ^= value >>> 13;
  return value >>> 0;
}

function totalOf(rows) {
  const total = rows[0].y.map(()=>0);
  for (const row of rows) row.y.forEach((value,index)=>{total[index]+=value;});
  return total;
}

function numericPredictor(rows) {
  const sorted = [...rows].sort((a,b)=>a.x-b.x);
  const total = totalOf(rows), mean = total.map(value=>value/rows.length);
  const left = total.map(()=>0);
  let best = -Infinity, threshold = null, low = mean, high = mean;
  for (let i=0;i<sorted.length-1;i++) {
    sorted[i].y.forEach((value,index)=>{left[index]+=value;});
    const count = i+1, remaining = sorted.length-count;
    if (count<5 || remaining<5 || sorted[i].x === sorted[i+1].x) continue;
    const gain = left.reduce((sum,value,index)=>sum+value*value/count+(total[index]-value)**2/remaining,0);
    if (gain>best) {
      best = gain;
      threshold = sorted[i].x/2+sorted[i+1].x/2;
      low = left.map(value=>value/count);
      high = total.map((value,index)=>(value-left[index])/remaining);
    }
  }
  return x => threshold === null ? mean : x<=threshold ? low : high;
}

function categoricalPredictor(rows) {
  const mean = totalOf(rows).map(value=>value/rows.length);
  const groups = new Map();
  for (const row of rows) {
    const key = valueKey(row.x);
    if (!groups.has(key)) groups.set(key,{count:0,sum:mean.map(()=>0)});
    const group = groups.get(key);
    group.count++;
    row.y.forEach((value,index)=>{group.sum[index]+=value;});
  }
  return x => {
    const group = groups.get(valueKey(x));
    return group ? group.sum.map((value,index)=>(value+5*mean[index])/(group.count+5)) : mean;
  };
}

export function evaluateParameter(rows,categoricalTarget=false) {
  if (rows.length<20) return {reason:'Fewer than 20 usable trials'};
  const counts = new Map();
  rows.forEach(row=>counts.set(valueKey(row.x),(counts.get(valueKey(row.x)) || 0)+1));
  if (counts.size<2) return {reason:'Constant recorded value'};
  const numeric = rows.every(row=>typeof row.x === 'number' && Number.isFinite(row.x));
  if (!numeric && [...counts.values()].filter(count=>count>=5).length<2) return {reason:'Too few repeated categories'};
  let normalized;
  if (categoricalTarget) {
    const classes = [...new Set(rows.map(row=>valueKey(row.y)))];
    if (classes.length<2) return {reason:'No target variation'};
    normalized = rows.map(row=>({...row,y:classes.map(value=>value===valueKey(row.y) ? 1 : 0)}));
  } else {
    const scale = Math.max(...rows.map(row=>Math.abs(row.y)));
    if (!scale) return {reason:'No metric variation'};
    const average = rows.reduce((sum,row)=>sum+row.y/scale,0)/rows.length;
    if (rows.every(row=>Math.abs(row.y/scale-average)<1e-14)) return {reason:'No metric variation'};
    normalized = rows.map(row=>({...row,y:[row.y/scale]}));
  }
  normalized.sort((a,b)=>hash(a.id)-hash(b.id) || a.id.localeCompare(b.id));
  let error = 0, baseline = 0, improved = 0;
  const folds = [];
  for (let fold=0;fold<5;fold++) {
    const train = normalized.filter((_,index)=>index%5!==fold), test = normalized.filter((_,index)=>index%5===fold);
    const mean = totalOf(train).map(value=>value/train.length);
    const predict = numeric ? numericPredictor(train) : categoricalPredictor(train);
    let foldError = 0, foldBaseline = 0;
    for (const row of test) {
      const prediction = predict(row.x);
      row.y.forEach((value,index)=>{
        foldError += (value-prediction[index])**2;
        foldBaseline += (value-mean[index])**2;
      });
    }
    error+=foldError;
    baseline+=foldBaseline;
    if (foldError<foldBaseline) improved++;
    folds.push(foldBaseline>0 ? 1-foldError/foldBaseline : null);
  }
  return {gain:1-error/baseline,improved,folds,used:rows.length,distinct:counts.size,type:numeric ? 'Numeric split' : 'Category means'};
}

export function rankParameters(trials,metric) {
  const started = performance.now();
  if (!Object.hasOwn(RESULTS,metric) && !/^(config|result)\//.test(metric)) throw new Error('Choose a supported metric.');
  const measured = trials.filter(trial=>valueOf(trial,metric) !== undefined && valueOf(trial,metric) !== null && (!['mse','r2'].includes(metric) || trial.status === 'complete'));
  const categorical = !measured.every(trial=>typeof valueOf(trial,metric) === 'number' && Number.isFinite(valueOf(trial,metric)));
  const targetCounts = new Map();
  measured.forEach(trial=>{const key=valueKey(valueOf(trial,metric));targetCounts.set(key,(targetCounts.get(key) || 0)+1);});
  const targetSparse = categorical && [...targetCounts.values()].filter(count=>count>=5).length<2;
  const fields = parameterFields(trials), patterns = new Map(), skipped = new Map();
  for (const field of fields) {
    const values = measured.map(trial=>valueOf(trial,field));
    const key = JSON.stringify(values.map(value=>value === undefined ? ['absent'] : ['value',value]));
    if (patterns.has(key)) { patterns.get(key).fields.push(field); continue; }
    if (field === metric || measured.length && values.every((value,index)=>valueKey(value) === valueKey(valueOf(measured[index],metric)))) {
      skipped.set('Target or identical values',(skipped.get('Target or identical values') || 0)+1);
      continue;
    }
    const rows = measured.flatMap((trial,index)=>values[index] === undefined || values[index] === null ? [] : [{id:trial.id,x:values[index],y:valueOf(trial,metric)}]);
    patterns.set(key,{fields:[field],...(targetSparse ? {reason:'Too few repeated target values'} : evaluateParameter(rows,categorical))});
  }
  const rankings = [];
  for (const result of patterns.values()) {
    if (result.reason) skipped.set(result.reason,(skipped.get(result.reason) || 0)+result.fields.length);
    else rankings.push({...result,coverage:result.used/measured.length});
  }
  rankings.sort((a,b)=>b.gain-a.gain || b.used-a.used || a.fields[0].localeCompare(b.fields[0]));
  const seconds = (performance.now()-started)/1000;
  console.info(`parameter ranking exit seconds=${seconds.toFixed(3)} fields=${fields.length}`);
  return {rankings,skipped:Object.fromEntries(skipped),measured:measured.length,total:trials.length,fields:fields.length,categorical,seconds};
}
