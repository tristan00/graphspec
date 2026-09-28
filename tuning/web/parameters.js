export const RESULTS = {number:'Trial order',mse:'Validation MSE',r2:'Validation R²',seconds:'Trial duration',parameters:'Model parameter count',status:'Status',selector:'Chosen by'};
export const METRICS = {mse:'Validation MSE ↓',r2:'Validation R² ↑',seconds:'Trial duration',parameters:'Model parameter count'};
const configs = new WeakMap();
const outcomes = new WeakMap();

export function flattenConfig(object, prefix = '') {
  const result = Object.create(null);
  for (const [name,value] of Object.entries(object || {})) {
    const key = prefix ? `${prefix}/${name}` : name;
    if (value !== null && typeof value === 'object' && !Array.isArray(value)) Object.assign(result,flattenConfig(value,key));
    else result[key] = value;
  }
  return result;
}

export function valueOf(trial, field) {
  if (field.startsWith('result/')) {
    if (!outcomes.has(trial)) outcomes.set(trial,flattenConfig(trial.metrics));
    return outcomes.get(trial)[field.slice(7)];
  }
  if (!field.startsWith('config/')) return Object.hasOwn(RESULTS,field) ? trial[field] : undefined;
  if (!configs.has(trial)) configs.set(trial,flattenConfig(trial.config));
  return configs.get(trial)[field.slice(7)];
}

export function fieldName(field) {
  return RESULTS[field] || field.replace(/^(config|result)\//,'');
}

export function resultFields(trials) {
  const keys = new Set();
  for (const trial of trials) {
    if (!outcomes.has(trial)) outcomes.set(trial,flattenConfig(trial.metrics));
    Object.keys(outcomes.get(trial)).filter(key=>key !== 'status').forEach(key=>keys.add(`result/${key}`));
  }
  return [...keys].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));
}

export function parameterFields(trials) {
  const keys = new Set();
  for (const trial of trials) {
    if (!configs.has(trial)) configs.set(trial,flattenConfig(trial.config));
    Object.keys(configs.get(trial)).forEach(key => keys.add(`config/${key}`));
  }
  return [...keys].sort((a,b) => a.localeCompare(b,undefined,{numeric:true}));
}

export function displayValue(value) {
  if (value === undefined) return 'Not used';
  if (value === null) return 'null';
  return typeof value === 'string' ? value || '""' : JSON.stringify(value);
}

export function valueKey(value) {
  return value === undefined ? 'missing' : JSON.stringify(value);
}

export function compareValues(left, right, direction = 'asc') {
  if (left === undefined || right === undefined) return left === right ? 0 : left === undefined ? 1 : -1;
  if (left === null || right === null) return left === right ? 0 : left === null ? 1 : -1;
  const order = typeof left === 'number' && typeof right === 'number' ? left-right : displayValue(left).localeCompare(displayValue(right),undefined,{numeric:true});
  return order * (direction === 'desc' ? -1 : 1);
}

export function matchesFilter(trial, condition) {
  const value = valueOf(trial,condition.field);
  if (condition.op === 'present') return value !== undefined;
  if (condition.op === 'missing') return value === undefined;
  if (value === undefined) return false;
  if (condition.op === 'is') return valueKey(value) === condition.value;
  if (condition.op === 'eq' || condition.op === 'ne') {
    const equal = typeof value === 'number'
      ? condition.value.trim() !== '' && Number.isFinite(Number(condition.value)) && value === Number(condition.value)
      : displayValue(value) === condition.value;
    return condition.op === 'eq' ? equal : !equal;
  }
  if (condition.op === 'contains') return displayValue(value).toLowerCase().includes(condition.value.toLowerCase());
  const target = condition.value.trim() === '' ? NaN : Number(condition.value);
  if (typeof value !== 'number' || !Number.isFinite(target)) return false;
  return {lt:value<target,lte:value<=target,gt:value>target,gte:value>=target}[condition.op] === true;
}

export function groupedResults(trials, field, metric) {
  const groups = new Map();
  for (const trial of trials) {
    const value = valueOf(trial,field), key = valueKey(value);
    if (!groups.has(key)) groups.set(key,{key,value,trials:[],scores:[]});
    const group = groups.get(key);
    group.trials.push(trial);
    const measured = valueOf(trial,metric);
    if (Number.isFinite(measured) && (!['mse','r2'].includes(metric) || trial.status === 'complete')) group.scores.push(measured);
  }
  return [...groups.values()].sort((a,b) => compareValues(a.value,b.value)).map(group => {
    group.scores.sort((a,b) => a-b);
    const length = group.scores.length, mid = Math.floor(length/2);
    return {...group,count:group.trials.length,scored:length,best:length ? group.scores[metric === 'r2' ? length-1 : 0] : null,
      median:length ? length%2 ? group.scores[mid] : (group.scores[mid-1]+group.scores[mid])/2 : null};
  });
}
