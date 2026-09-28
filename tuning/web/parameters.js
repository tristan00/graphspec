const FIXED = {number:'Trial order',seconds:'Trial duration',parameters:'Model parameter count',status:'Status',selector:'Chosen by'};
export const RESULTS = {...FIXED};
export const METRICS = {seconds:'Trial duration',parameters:'Model parameter count'};
export const MEASURES = [];
const BETTER = {};
const SHORT = {};

export function setMeasures(measures) {
  for (const key of Object.keys(BETTER)) { delete RESULTS[key]; delete BETTER[key]; delete SHORT[key]; }
  for (const key of Object.keys(METRICS)) delete METRICS[key];
  MEASURES.splice(0,MEASURES.length,...measures);
  for (const {key,label,short,better} of measures) {
    RESULTS[key] = label;
    METRICS[key] = `${label} ${better === 'higher' ? '↑' : '↓'}`;
    BETTER[key] = better;
    SHORT[key] = short;
  }
  Object.assign(METRICS,{seconds:FIXED.seconds,parameters:FIXED.parameters});
}

export const isMeasure = field => Object.hasOwn(BETTER,field);
export const higherIsBetter = field => BETTER[field] === 'higher';
export const shortName = field => SHORT[field] ?? field;
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
  if (!field.startsWith('config/')) {
    if (!Object.hasOwn(RESULTS,field)) return undefined;
    return Object.hasOwn(trial,field) ? trial[field] : trial.scores?.[field];
  }
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
    if (Number.isFinite(measured) && (!isMeasure(metric) || trial.status === 'complete')) group.scores.push(measured);
  }
  return [...groups.values()].sort((a,b) => compareValues(a.value,b.value)).map(group => {
    group.scores.sort((a,b) => a-b);
    const length = group.scores.length, mid = Math.floor(length/2);
    return {...group,count:group.trials.length,scored:length,best:length ? group.scores[higherIsBetter(metric) ? length-1 : 0] : null,
      median:length ? length%2 ? group.scores[mid] : (group.scores[mid-1]+group.scores[mid])/2 : null};
  });
}
