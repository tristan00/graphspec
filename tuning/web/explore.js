import {METRICS, valueOf, fieldName, displayValue, valueKey, compareValues, matchesFilter, groupedResults} from '/parameters.js';
import {hierarchyEntries} from '/discovery.js';

const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const numeric = value => typeof value === 'number' && Number.isFinite(value);
const score = value => !numeric(value) ? '—' : value === 0 ? '0' : Math.abs(value) < .0001 || Math.abs(value) >= 100000 ? value.toExponential(3) : value.toLocaleString('en-US',{maximumSignificantDigits:5});
const metricName = field => METRICS[field] || fieldName(field);
const OPS = {eq:'equals',ne:'does not equal',lt:'<',lte:'≤',gt:'>',gte:'≥',contains:'contains',present:'is used',missing:'is not used',is:'equals'};

export function filterRows(trials,state) {
  const query = state.query.trim().toLowerCase();
  return trials.filter(trial => (state.status === 'all' || trial.status === state.status) && state.conditions.every(condition => matchesFilter(trial,condition)) && (!query || `${trial.number} ${trial.id} ${trial.description} ${trial.selector} ${JSON.stringify(trial.config)}`.toLowerCase().includes(query)));
}

export function sortedRows(trials,state) {
  return filterRows(trials,state).sort((a,b) => compareValues(valueOf(a,state.sort),valueOf(b,state.sort),state.direction) || a.number-b.number);
}

const pickers = new Map();

export function pickerState(id) {
  if (!pickers.has(id)) pickers.set(id,{path:[],query:'',limit:80});
  return pickers.get(id);
}

export function resetPickers() {
  pickers.clear();
}

export function pickerOptions(fields,id,query = '',configOnly = false) {
  const current = pickerState(id);
  current.query = query;
  const entries = hierarchyEntries(fields.filter(field=>!configOnly || field.startsWith('config/')),current.path,query);
  const crumbs = [[],...current.path.map((_,index)=>current.path.slice(0,index+1))].map(path=>`<button type="button" data-picker-path="${esc(JSON.stringify(path))}" data-picker-target="${id}"${path.length === current.path.length ? ' aria-current="location"' : ''}>${esc(path.at(-1) || 'All fields')}</button>`).join('<span aria-hidden="true">/</span>');
  const folders = entries.folders.map(folder=>`<button type="button" class="picker-folder" data-picker-path="${esc(JSON.stringify(folder.path))}" data-picker-target="${id}"><span>▸ ${esc(folder.name)}</span><small>${folder.count}</small></button>`).join('');
  const leaves = entries.leaves.slice(0,current.limit).map(entry=>`<button type="button" data-field-choice="${esc(entry.field)}" data-picker-target="${id}"><span>${esc(query.trim() ? fieldName(entry.field) : entry.field.startsWith('config/') ? entry.name : fieldName(entry.field))}</span>${query.trim() ? `<small>${entry.field.startsWith('config/') ? 'Setting' : 'Result'}</small>` : ''}</button>`).join('');
  return `<div class="picker-breadcrumb" aria-label="Parameter location">${crumbs}</div><div class="picker-list">${folders}${leaves}${!entries.total ? '<p class="picker-count">No matching fields in this branch.</p>' : ''}</div>${entries.leaves.length>current.limit ? `<button type="button" class="picker-more" data-picker-more="${id}">Show ${Math.min(80,entries.leaves.length-current.limit)} more</button>` : ''}<p class="picker-count">${entries.total} fields${current.path.length ? ' in this branch' : ''}${current.path.length && query.trim() ? ` · <button type="button" data-picker-path="[]" data-picker-target="${id}">Search all fields</button>` : ''}</p>`;
}

export function fieldPicker(fields,id,selected,title,configOnly = false) {
  const current = pickerState(id);
  return `<div class="field-control"><label id="${id}-label">${title}</label><details class="field-picker" id="${id}" data-config-only="${configOnly}"><summary aria-labelledby="${id}-label ${id}-chosen"><span id="${id}-chosen">${selected ? esc(fieldName(selected)) : 'Choose a parameter…'}</span><span aria-hidden="true">⌄</span></summary><div class="picker-menu"><input type="search" id="${id}-search" data-picker-search="${id}" aria-label="Find ${title.toLowerCase()}" value="${esc(current.query)}" placeholder="Search ${current.path.length ? 'this branch' : 'all fields'}…" autocomplete="off"><div class="picker-options">${pickerOptions(fields,id,current.query,configOnly)}</div></div></details></div>`;
}

export function filterToolbar(data,state,fields,draft) {
  const choices = draft.field ? [...new Set(data.trials.map(trial => valueOf(trial,draft.field)).filter(value => value !== undefined).map(displayValue))].sort((a,b) => a.localeCompare(b,undefined,{numeric:true})) : [];
  return `<div class="filters"><input type="search" id="search" aria-label="Search trials" placeholder="Search trial records…" value="${esc(state.query)}"><label class="filter-label">Status<select id="status"><option value="all">All statuses</option>${['complete','running','pending','failed','timeout','interrupted'].map(status => `<option value="${status}"${state.status === status ? ' selected' : ''}>${status}</option>`).join('')}</select></label><button type="button" class="button quiet" data-action="clear-filters">Reset filters</button></div><div class="condition-list">${state.conditions.map((condition,index) => `<span class="condition-chip">${esc(fieldName(condition.field))} <strong>${OPS[condition.op]}${['present','missing'].includes(condition.op) ? '' : ` ${esc(condition.op === 'is' ? displayValue(JSON.parse(condition.value)) : condition.value)}`}</strong><button class="button quiet" data-remove-condition="${index}" aria-label="Remove filter ${index+1}">×</button></span>`).join('')}</div><form id="condition-form" class="condition-editor">${fieldPicker(fields,'filter-field',draft.field,'Filter parameter')}<label class="field-control">Condition<select id="filter-op">${Object.entries(OPS).filter(([key]) => key !== 'is').map(([key,text]) => `<option value="${key}"${draft.op === key ? ' selected' : ''}>${text}</option>`).join('')}</select></label><label class="field-control condition-value">Value<input id="filter-value" type="text" list="parameter-values" value="${esc(draft.value)}"${['present','missing'].includes(draft.op) ? ' disabled' : ''} autocomplete="off" placeholder="Value or number"><datalist id="parameter-values">${choices.map(value => `<option value="${esc(value)}"></option>`).join('')}</datalist></label><button type="submit" class="button"${draft.field ? '' : ' disabled'}>Add filter</button><span class="small muted">All filters apply to the graph and trials.</span></form>`;
}

export function analysisView(data,state,fields,draft) {
  const rows = filterRows(data.trials,state);
  const finished = data.trials.filter(trial => !['running','pending'].includes(trial.status)).length;
  const complete = data.trials.filter(trial => trial.status === 'complete').length;
  const active = data.trials.find(trial => trial.status === 'running');
  const status = data.runner?.status || 'unknown';
  return `<div class="run-line"><span class="badge ${esc(status)}">${esc(status)}</span><span>${data.trials.length} trials · ${complete} complete · ${finished-complete} unsuccessful</span>${active ? `<button class="text-button" data-trial="${esc(active.id)}">Trial ${active.number}: ${esc(active.progress?.phase || 'starting')}</button>` : ''}<details id="run-context" class="run-context"><summary>Run details</summary><dl class="settings-grid"><div><dt>Trial budget</dt><dd>${data.plan.seconds}s</dd></div><div><dt>Workers</dt><dd>${data.plan.workers ?? '—'}</dd></div><div><dt>Training / holdout rows</dt><dd>${data.plan.train_rows} / ${data.plan.holdout_rows}</dd></div><div><dt>Active epoch</dt><dd>${active ? `${active.progress?.epoch_seconds?.length || 0} / ${active.config.epochs}` : 'No active worker'}</dd></div><div><dt>Bank</dt><dd>${esc(data.plan.bank)}</dd></div><div><dt>Run directory</dt><dd>${esc(data.path)}</dd></div></dl></details></div>${data.runner?.error ? `<div class="notice">${esc(data.runner.error)}</div>` : ''}${data.stop_requested ? '<div class="notice">The runner has received a stop request.</div>' : ''}<section class="panel analysis-panel"><div class="analysis-controls">${fieldPicker(fields,'axis-field',state.x,'Plot by')}<label class="field-control">Measure<select id="chart-metric" aria-label="Chart metric">${Object.entries({...METRICS,[state.metric]:metricName(state.metric)}).map(([key,text]) => `<option value="${key}"${state.metric === key ? ' selected' : ''}>${text}</option>`).join('')}</select></label><button class="button" data-action="inspect-rows">Inspect matching trials →</button></div>${filterToolbar(data,state,fields,draft)}<div id="analysis-results">${analysisResults(rows,state)}</div></section>`;
}

export function analysisResults(rows,state) {
  if (!state.x) return '<div class="analysis-empty"><h2>What do you want to investigate?</h2><p>Choose any saved parameter in <strong>Plot by</strong> to graph its results and group trials by its values. Choose Trial order to see the search over time.</p></div>';
  const groups = groupedResults(rows,state.x,state.metric);
  const metric = metricName(state.metric);
  return `<div class="analysis-heading"><h2>${esc(metric)} by ${esc(fieldName(state.x))}</h2><p>${rows.length} matching trials · values come directly from saved configurations and results</p></div>${plot(rows,state)}<div class="group-heading"><h3>Results grouped by ${esc(fieldName(state.x))}</h3><button class="button" data-action="export-groups"${rows.length ? '' : ' disabled'}>Export grouped CSV</button></div><p class="analysis-note">Each row is one exact value. Counts include unsuccessful trials; score summaries use recorded ${['mse','r2'].includes(state.metric) ? 'completed' : ''} results. Other parameters may differ.</p><div class="table-wrap grouped-results"><table aria-label="Results by parameter"><thead><tr><th scope="col">${esc(fieldName(state.x))}</th><th scope="col">Trials</th><th scope="col">With measure</th><th scope="col">${state.metric === 'r2' ? 'Maximum' : 'Minimum'}</th><th scope="col">Median</th><th scope="col">Inspect</th></tr></thead><tbody>${groups.map(group => `<tr><td class="parameter-value">${esc(displayValue(group.value))}</td><td>${group.count}</td><td>${group.scored}</td><td>${score(group.best)}</td><td>${score(group.median)}</td><td><button class="text-button" data-group-value="${esc(group.key)}">View ${group.count} trials</button></td></tr>`).join('') || '<tr><td colspan="6">No matching trials. Change or reset the filters.</td></tr>'}</tbody></table></div>`;
}

function plot(rows,state) {
  const measured = rows.filter(trial => numeric(valueOf(trial,state.metric)) && (!['mse','r2'].includes(state.metric) || trial.status === 'complete'));
  const trials = measured.filter(trial => valueOf(trial,state.x) !== undefined);
  const missing = measured.length-trials.length;
  const note = `${trials.length} plotted · ${rows.length-measured.length} without a recorded measure · ${missing} without this parameter. Select a point to inspect; arrow keys move between trials.`;
  if (!trials.length) return `<div class="chart-empty"><span>${!rows.length ? 'No matching trials.' : !measured.length ? 'No recorded measures for these trials yet.' : 'This parameter is not used by any measured trial.'}</span></div><p class="analysis-note">${note}</p>`;
  const values = trials.map(trial => valueOf(trial,state.x));
  const continuous = values.every(numeric);
  const categories = [...new Map(values.map(value => [valueKey(value),value])).values()].sort((a,b) => compareValues(a,b));
  const positions = new Map(categories.map((value,index) => [valueKey(value),index]));
  const width = continuous ? 900 : Math.max(900,categories.length*92+110);
  const left = 74, right = width-28, top = 24, bottom = 250;
  let low = Math.min(...trials.map(trial => valueOf(trial,state.metric))), high = Math.max(...trials.map(trial => valueOf(trial,state.metric)));
  const pad = (high-low || Math.max(Math.abs(high)*.1,.01))*.1;
  low = state.metric === 'r2' ? low-pad : Math.max(0,low-pad);
  high += pad;
  const minX = continuous ? Math.min(...values) : 0, maxX = continuous ? Math.max(...values) : categories.length-1;
  const x = value => minX === maxX ? (left+right)/2 : left+((continuous ? value : positions.get(valueKey(value)))-minX)/(maxX-minX)*(right-left);
  const y = value => bottom-(value-low)/(high-low)*(bottom-top);
  const grid = Array.from({length:5},(_,index) => { const value=low+(high-low)*index/4; return `<line x1="${left}" y1="${y(value)}" x2="${right}" y2="${y(value)}" stroke="var(--line)"/><text x="${left-10}" y="${y(value)+4}" text-anchor="end">${esc(score(value))}</text>`; }).join('');
  const ticks = continuous ? categories.length <= 10 ? categories : Array.from({length:6},(_,index) => minX+(maxX-minX)*index/5) : categories;
  const labels = ticks.map(value => `<text x="${x(value)}" y="${bottom+24}" text-anchor="middle"><title>${esc(displayValue(value))}</title>${esc(continuous ? score(value) : displayValue(value).length>17 ? displayValue(value).slice(0,16)+'…' : displayValue(value))}</text>`).join('');
  let path = '';
  if (state.x === 'number' && ['mse','r2'].includes(state.metric)) {
    let best = state.metric === 'r2' ? -Infinity : Infinity;
    for (const trial of [...trials].sort((a,b) => a.number-b.number)) {
      best = state.metric === 'r2' ? Math.max(best,valueOf(trial,state.metric)) : Math.min(best,valueOf(trial,state.metric));
      path += path ? `H${x(trial.number)}V${y(best)}` : `M${x(trial.number)},${y(best)}`;
    }
  }
  const points = trials.map((trial,index) => `<circle class="point" role="button" tabindex="${index ? -1 : 0}" data-trial="${esc(trial.id)}" aria-label="Trial ${trial.number}, ${esc(fieldName(state.x))} ${esc(displayValue(valueOf(trial,state.x)))}, ${esc(metricName(state.metric))} ${score(valueOf(trial,state.metric))}" cx="${x(valueOf(trial,state.x))}" cy="${y(valueOf(trial,state.metric))}" r="5" fill="var(--green)" fill-opacity=".65"><title>Trial ${trial.number} · ${esc(displayValue(valueOf(trial,state.x)))} · ${score(valueOf(trial,state.metric))}</title></circle>`).join('');
  return `<div class="parameter-chart"><svg style="min-width:${width}px" viewBox="0 0 ${width} 316" role="group" aria-label="${esc(metricName(state.metric))} by ${esc(fieldName(state.x))}">${grid}${labels}${path ? `<path d="${path}" fill="none" stroke="var(--plot-line)" stroke-width="2"/>` : ''}${points}<text x="${(left+right)/2}" y="307" text-anchor="middle">${esc(fieldName(state.x))}</text></svg></div><p class="analysis-note">${note}${path ? ' Line: best so far within these filters.' : ''} Scroll horizontally when the graph does not fit.</p>`;
}
