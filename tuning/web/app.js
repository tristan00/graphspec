import {RESULTS, METRICS, MEASURES, setMeasures, isMeasure, higherIsBetter, valueOf, fieldName, parameterFields, resultFields, displayValue, groupedResults} from '/parameters.js';
import {analysisView, analysisResults, filterRows, sortedRows, fieldPicker, pickerOptions, pickerState, resetPickers, filterToolbar} from '/explore.js';
import {discoveryView, discoveryResults} from '/discover-view.js';
import {currentRunView, botorchView, forestView, tpeView} from '/run-view.js';

const $ = (selector, root = document) => root.querySelector(selector);
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const finite = value => typeof value === 'number' && Number.isFinite(value);
const number = value => finite(value) ? new Intl.NumberFormat('en-US', {maximumFractionDigits: 0}).format(value) : '—';
const score = value => !finite(value) ? '—' : value === 0 ? '0' : Math.abs(value) < .0001 || Math.abs(value) >= 100000 ? value.toExponential(3) : value.toLocaleString('en-US', {maximumSignificantDigits: 5});
const duration = value => !finite(value) ? '—' : value < 60 ? `${value.toFixed(1)}s` : value < 3600 ? `${(value / 60).toFixed(1)}m` : `${(value / 3600).toFixed(1)}h`;
const label = value => String(value ?? 'unknown').replaceAll('_', ' ');
const SELECTORS = {random:'Random',forest:'SMAC',botorch:'BoTorch',tpe:'TPE'};
const badge = status => `<span class="badge ${esc(status)}">${esc(label(status))}</span>`;
const empty = (title, text, action = '') => `<div class="empty"><div class="empty-symbol" aria-hidden="true">◫</div><h2>${esc(title)}</h2><p>${esc(text)}</p>${action}</div>`;
const trialButton = (trial, text = `Trial ${String(trial.number).padStart(3, '0')}`) => `<button class="trial-link" data-trial="${esc(trial.id)}">${esc(text)}</button>`;
const fields = object => Object.entries(object || {}).map(([key, value]) => `<div><dt>${esc(key)}</dt><dd>${esc(value ?? '—')}</dd></div>`).join('');
const OPTIMIZER_VIEWS = ['forest','botorch','tpe'];
const state = {run:'', view:'overview', proposal:'', query:'', status:'all', sort:'number', direction:'desc', page:1, metric:'mse', target:'', x:'', columns:[], conditions:[], selected:[], differences:false};
let availableFields = [];
let filterDraft = {field:'',op:'eq',value:''};
let data = null;
let catalogue = null;
let request = null;
let detailRequest = null;
let detailData = null;
let detailId = null;
let detailTab = 'summary';
let poll = null;
let toastTimer = null;
let discovery = {key:'',result:null,busy:false,error:'',query:'',positive:true,limit:30,at:''};
let rankingWorker = null;
let defaultsApplied = '';

function readLocation() {
  const params = new URLSearchParams(location.hash.slice(1));
  state.run = params.get('run') || '';
  state.view = ['current','forest','botorch','tpe','overview','discover','trials','compare'].includes(params.get('view')) ? params.get('view') : 'overview';
  state.proposal = params.get('proposal') || '';
  state.query = params.get('q') || '';
  state.status = ['all','complete','running','pending','failed','timeout','interrupted'].includes(params.get('status')) ? params.get('status') : 'all';
  state.sort = Object.hasOwn(RESULTS,params.get('sort')) || /^(config|result|series)\//.test(params.get('sort') || '') || ['mse','r2'].includes(params.get('sort')) ? params.get('sort') : 'number';
  state.direction = params.get('dir') === 'asc' ? 'asc' : 'desc';
  state.page = Math.max(1, Math.min(1000000, Number.parseInt(params.get('page'), 10) || 1));
  state.metric = Object.hasOwn(RESULTS,params.get('metric')) || /^(config|result|series)\//.test(params.get('metric') || '') || params.get('metric') === 'r2' ? params.get('metric') : 'mse';
  state.x = params.get('x') || '';
  state.target = params.get('target') || '';
  state.columns = [...new Set(params.getAll('column'))].filter(field => /^(config|result|series)\//.test(field));
  state.conditions = params.getAll('where').flatMap(text => {
    try {
      const condition = JSON.parse(text);
      if (condition.op === 'is') JSON.parse(condition.value);
      return typeof condition.field === 'string' && ['eq','ne','is','lt','lte','gt','gte','contains','present','missing'].includes(condition.op) && typeof condition.value === 'string' ? [condition] : [];
    } catch { return []; }
  });
  if (params.get('family') && params.get('family') !== 'all') state.conditions.push({field:'config/model/view',op:'eq',value:params.get('family')});
  state.selected = [...new Set(params.getAll('compare'))].slice(0, 4);
  state.differences = params.get('diff') === '1';
}

function saveLocation(push = false) {
  const params = new URLSearchParams();
  if (state.run) params.set('run', state.run);
  params.set('view', state.view);
  if (state.proposal) params.set('proposal', state.proposal);
  if (state.query) params.set('q', state.query);
  if (state.status !== 'all') params.set('status', state.status);
  if (state.x) params.set('x', state.x);
  if (state.target) params.set('target', state.target);
  state.columns.forEach(field => params.append('column',field));
  state.conditions.forEach(condition => params.append('where',JSON.stringify(condition)));
  params.set('sort', state.sort);
  params.set('dir', state.direction);
  if (state.page > 1) params.set('page', state.page);
  if (state.metric !== 'mse') params.set('metric', state.metric);
  for (const id of state.selected) params.append('compare', id);
  if (state.differences) params.set('diff', '1');
  const hash = '#' + params.toString();
  if (hash !== location.hash) history[push ? 'pushState' : 'replaceState'](null, '', hash);
}

async function api(path, signal) {
  const deadline = new AbortController();
  const timer = setTimeout(() => deadline.abort(new DOMException('The server did not respond within 10 seconds.', 'TimeoutError')),10000);
  try {
    const response = await fetch(path, {signal:AbortSignal.any([signal,deadline.signal]), cache:'no-store'});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `Request failed (${response.status}).`);
    return body;
  } finally {
    clearTimeout(timer);
  }
}

function toast(message) {
  clearTimeout(toastTimer);
  $('#toast').textContent = message;
  $('#toast').hidden = false;
  toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 3500);
}

function showError(message) {
  $('#error').textContent = message;
  $('#error').hidden = false;
}

function schedule() {
  clearTimeout(poll);
  if ($('#live').checked && !document.hidden) poll = setTimeout(() => refresh(), 5000);
}

async function refreshCatalog(signal) {
  catalogue = await api('/api/runs', signal);
  $('#source').textContent = catalogue.root;
  const select = $('#run');
  const options = catalogue.runs.map(run => `<option value="${esc(run.id)}">${esc(run.id)}${run.error ? ' · unreadable' : ''}</option>`).join('');
  select.innerHTML = options || '<option value="">No runs found</option>';
  select.disabled = !catalogue.runs.length;
  if (!state.run && catalogue.runs.length) {
    state.run = (catalogue.runs.find(run => run.status === 'running') || catalogue.runs[0]).id;
    saveLocation();
  }
  if (state.run && !catalogue.runs.some(run => run.id === state.run)) {
    select.insertAdjacentHTML('afterbegin', `<option value="${esc(state.run)}">${esc(state.run)} · unavailable</option>`);
  }
  select.value = state.run;
}

async function refresh(includeCatalog = false) {
  if (request) request.abort();
  const controller = new AbortController();
  request = controller;
  clearTimeout(poll);
  $('#refresh').disabled = true;
  $('#sync').textContent = 'Updating…';
  try {
    if (includeCatalog || !catalogue || !state.run) await refreshCatalog(controller.signal);
    if (controller !== request) return;
    if (!state.run) {
      data = null;
      render();
    } else {
      const identity = state.run;
      const result = await api('/api/run?' + new URLSearchParams({id:identity}), controller.signal);
      if (controller !== request || identity !== state.run) return;
      data = result;
      setMeasures(data.measures);
      if (defaultsApplied !== identity) {
        defaultsApplied = identity;
        if (!state.columns.length) state.columns = [...data.columns];
      }
      if (OPTIMIZER_VIEWS.includes(state.view)) {
        const optimizer = await api('/api/optimizer?' + new URLSearchParams({id:identity}), controller.signal);
        if (controller !== request || identity !== state.run) return;
        data.optimizer = optimizer;
      }
      availableFields = [...Object.keys(RESULTS),...resultFields(data.trials),...parameterFields(data.trials)];
      state.selected = state.selected.filter(id => data.trials.some(trial => trial.id === id));
      saveLocation();
      render();
      if ($('#detail').open && detailId) loadDetail(detailId, true);
    }
    $('#error').hidden = true;
    $('#sync').textContent = `Updated ${new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'})}`;
  } catch (error) {
    if (error.name === 'AbortError' || controller !== request) return;
    showError(`${data ? 'Showing the last successful update. ' : ''}${error.message} Use Refresh to retry.`);
    $('#sync').textContent = data ? 'Update failed · data may be stale' : 'Connection failed';
    if (!data) $('#content').innerHTML = `<div class="panel">${empty('Run unavailable', 'The saved records could not be loaded. Choose another run or retry.', '<button class="button primary" data-action="retry">Retry connection</button>')}</div>`;
  } finally {
    if (controller === request) {
      request = null;
      $('#refresh').disabled = false;
      $('#content').setAttribute('aria-busy', 'false');
      schedule();
    }
  }
}

function switchRun(identity, push = true) {
  state.run = identity;
  state.selected = [];
  state.proposal = '';
  state.query = '';
  state.status = 'all';
  state.conditions = [];
  state.columns = [];
  if (!['number','mse','r2'].includes(state.sort)) state.sort = 'number';
  if (!['mse','r2'].includes(state.metric)) state.metric = 'mse';
  filterDraft = {field:'',op:'eq',value:''};
  resetPickers();
  state.page = 1;
  data = null;
  $('#detail').close();
  $('#error').hidden = true;
  saveLocation(push);
  render();
  refresh();
}

function navigate(view) {
  state.view = view;
  saveLocation(true);
  render();
  if (OPTIMIZER_VIEWS.includes(view) && data && !data.optimizer) refresh();
  $('#page-title').scrollIntoView({block:'nearest'});
}

function overview() {
  return analysisView(data,state,availableFields,filterDraft);
}

function syncDiscovery() {
  const key = JSON.stringify([state.run,state.target,state.query,state.status,state.conditions]);
  if (key === discovery.key) return;
  rankingWorker?.terminate();
  rankingWorker = null;
  discovery = {key,result:null,busy:false,error:'',query:'',positive:true,limit:30,at:''};
}

function rankParameters() {
  if (!data || !state.target) return;
  syncDiscovery();
  rankingWorker?.terminate();
  discovery.busy = true;
  discovery.error = '';
  const key = discovery.key;
  const at = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
  const finish = (worker,message) => {
    if (worker !== rankingWorker || key !== discovery.key) return;
    worker?.terminate();
    rankingWorker = null;
    discovery.busy = false;
    discovery.result = message.result || null;
    discovery.error = message.error || '';
    discovery.at = at;
    if (state.view === 'discover') render();
  };
  try {
    const worker = new Worker('/discovery-worker.js',{type:'module'});
    rankingWorker = worker;
    worker.onmessage = event=>finish(worker,event.data);
    worker.onerror = ()=>finish(worker,{error:'Parameter ranking could not finish.'});
    worker.postMessage({trials:filterRows(data.trials,state),metric:state.target,measures:MEASURES});
  } catch (error) {
    finish(rankingWorker,{error:error.message});
  }
  render();
}

function refreshPicker(picker) {
  const current = pickerState(picker.id);
  $('input',picker).value = current.query;
  $('input',picker).placeholder = `Search ${current.path.length ? 'this branch' : 'all fields'}…`;
  $('.picker-options',picker).innerHTML = pickerOptions(availableFields,picker.id,current.query,picker.dataset.configOnly === 'true');
}

function filtered() {
  return sortedRows(data.trials,state);
}

function selectionBar() {
  return state.selected.length ? `<div class="selection-bar"><strong>${state.selected.length} of 4 trials selected</strong><span class="small">Selections stay while you filter.</span><span class="spacer"></span><button class="button primary" data-view="compare">Compare trials →</button><button class="button quiet" data-action="clear-selection">Clear selection</button></div>` : '';
}

function trialsView() {
  return `${selectionBar()}<section class="panel trial-panel"><div class="panel-head"><div><h2>Trial explorer</h2><p>Add any saved parameter as a column or sort key. Select up to four trials to compare.</p></div><button id="export" class="button"${filtered().length ? '' : ' disabled'}>↓ Export CSV</button></div>${filterToolbar(data,state,availableFields,filterDraft)}<div class="table-controls">${fieldPicker(availableFields,'column-field','','Add column',true)}${fieldPicker(availableFields,'sort-field',state.sort,'Sort by')}<button class="button" id="sort-direction" aria-label="Reverse sort direction">${state.direction === 'asc' ? 'Ascending ↑' : 'Descending ↓'}</button><button class="button" data-view="overview">Graph these trials →</button></div><div class="column-chips">${state.columns.map((field,index) => `<span class="condition-chip">${esc(fieldName(field))}<button class="button quiet" data-remove-column="${index}" aria-label="Remove column ${esc(fieldName(field))}">×</button></span>`).join('')}</div><div id="trial-results">${trialResults()}</div></section>`;
}

function trialResults() {
  const rows = filtered();
  const pages = Math.max(1,Math.ceil(rows.length/25));
  state.page = Math.min(state.page,pages);
  const start = (state.page-1)*25;
  return `${rows.length ? trialTable(rows.slice(start,start+25), true) : empty(data.trials.length ? 'No matching trials' : 'No trials yet', data.trials.length ? 'Try another search or reset your filters.' : 'Trials will appear when the tuning runner proposes them.', data.trials.length ? '<button class="button" data-action="clear-filters">Reset filters</button>' : '')}<div class="table-footer"><span>${rows.length ? `${start+1}–${Math.min(start+25,rows.length)} of ${rows.length}` : '0'} trials${rows.length !== data.trials.length ? ` · ${data.trials.length} total` : ''}</span><div class="pagination"><button class="button" data-page="-1"${state.page === 1 ? ' disabled' : ''} aria-label="Previous page">←</button><span>Page ${state.page} of ${pages}</span><button class="button" data-page="1"${state.page === pages ? ' disabled' : ''} aria-label="Next page">→</button></div></div>`;
}

function trialTable(rows, sortable) {
  const columns = ['number','status','selector',...state.columns,'mse','r2','seconds','parameters'];
  const heading = key => `<th scope="col" aria-sort="${state.sort === key ? state.direction === 'asc' ? 'ascending' : 'descending' : 'none'}"><button class="sort-button${state.sort === key ? ' active' : ''}" data-sort="${esc(key)}">${esc(fieldName(key))}${state.sort === key ? state.direction === 'asc' ? ' ↑' : ' ↓' : ' ↕'}</button></th>`;
  const cell = (trial,key) => key === 'number' ? trialButton(trial) : key === 'status' ? badge(trial.status) : key === 'selector' ? esc(SELECTORS[trial.selector] || label(trial.selector)) : key === 'seconds' ? duration(trial.seconds) : key === 'parameters' ? number(trial.parameters) : isMeasure(key) ? score(valueOf(trial,key)) : esc(displayValue(valueOf(trial,key)));
  return `<div class="table-wrap"><table aria-label="All matching trials"><thead><tr><th scope="col">Compare</th>${columns.map(heading).join('')}</tr></thead><tbody>${rows.map(trial => `<tr><td><input type="checkbox" data-select="${esc(trial.id)}" aria-label="Compare trial ${trial.number}"${state.selected.includes(trial.id) ? ' checked' : ''}></td>${columns.map(key => `<td class="${key.startsWith('config/') ? 'parameter-value' : 'numeric'}" data-column="${esc(key)}">${cell(trial,key)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function flatten(value, prefix = '') {
  const result = {};
  for (const [key, entry] of Object.entries(value || {})) {
    const name = prefix ? `${prefix}/${key}` : key;
    if (entry && typeof entry === 'object' && !Array.isArray(entry)) Object.assign(result,flatten(entry,name));
    else result[name] = entry;
  }
  return result;
}

function compareView() {
  const trials = state.selected.map(id => data.trials.find(trial => trial.id === id)).filter(Boolean);
  if (!trials.length) return `<section class="panel">${empty('Put the choices side by side', 'Select up to four trials in the explorer to compare their scores, model settings, and graph parameters.', '<button class="button primary" data-view="trials">Choose trials →</button>')}</section>`;
  const configs = trials.map(trial => flatten(trial.config));
  const keys = [...new Set(configs.flatMap(config => Object.keys(config)))].sort();
  let visible = 0;
  const configRows = keys.map(key => {
    const values = configs.map(config => Object.hasOwn(config,key) ? JSON.stringify(config[key]) : undefined);
    const different = new Set(values).size > 1;
    if (state.differences && !different) return '';
    visible++;
    return `<tr${different ? ' class="different"' : ''}><th scope="row">${esc(key)}</th>${values.map(value => `<td>${esc(value ?? 'Not used')}</td>`).join('')}</tr>`;
  }).join('');
  const row = (title, fn) => `<tr><th scope="row">${title}</th>${trials.map(trial => `<td>${fn(trial)}</td>`).join('')}</tr>`;
  return `<section class="panel"><div class="panel-head"><div><h2>${trials.length} trial${trials.length === 1 ? '' : 's'}, side by side</h2><p>Changed parameters are highlighted. Missing settings are marked “Not used”.</p></div><button class="button" data-view="trials">Add trials</button></div><div class="filters"><label class="live-switch small"><input type="checkbox" id="differences"${state.differences ? ' checked' : ''}> Only differing parameters</label><button class="button quiet" data-action="clear-selection">Clear selection</button><button class="button" data-action="export-configs">Export configurations</button></div><div class="table-wrap"><table class="compare-table" aria-label="Trial comparison"><thead><tr><th scope="col">Result / parameter</th>${trials.map(trial => `<th scope="col"><div class="compare-heading">${trialButton(trial)}<button class="button quiet" data-remove="${esc(trial.id)}" aria-label="Remove trial ${trial.number}">×</button></div></th>`).join('')}</tr></thead><tbody>${row('Status', t => badge(t.status))}${MEASURES.map(measure => row(esc(METRICS[measure.key]), t => score(valueOf(t,measure.key)))).join('')}${row('Duration', t => duration(t.seconds))}${row('Parameters', t => number(t.parameters))}${row('Chosen by', t => esc(SELECTORS[t.selector] || label(t.selector)))}<tr class="group"><td colspan="${trials.length+1}">Configuration · ${visible} ${state.differences ? 'differing ' : ''}parameters</td></tr>${configRows || `<tr><td colspan="${trials.length+1}">No differing parameters in this selection.</td></tr>`}</tbody></table></div></section>`;
}

function render() {
  syncDiscovery();
  const disclosures = [...$('#content').querySelectorAll('details[id][open]')].map(element => element.id);
  const opened = [...document.querySelectorAll('.field-picker[open]')].map(element => ({id:element.id,query:$('input',element).value,scroll:$('.picker-list',element)?.scrollTop || 0}));
  const focus = document.activeElement;
  const focusId = focus?.id;
  const focusKey = ['trial','select','view','sort','page','remove','action','fieldChoice','pickerPath','discoverField'].find(key => focus?.dataset?.[key]);
  const focusValue = focusKey ? focus.dataset[focusKey] : null;
  const selection = ['search','text'].includes(focus?.type) ? [focus.selectionStart,focus.selectionEnd] : null;
  const tableScroll = $('.table-wrap')?.scrollLeft;
  document.querySelectorAll('nav [data-view]').forEach(button => {
    if (button.dataset.view === state.view) button.setAttribute('aria-current','page');
    else button.removeAttribute('aria-current');
  });
  $('#page-title').textContent = {current:'Current run',botorch:'Inside BoTorch',overview:'Analyze the search',discover:'Discover parameters',trials:'Explore the trials',compare:'Compare trials'}[state.view];
  $('#subtitle').textContent = {current:'What’s happening now, and how the run is progressing.',forest:'How the random forest ranked its candidates, and what it split on.',botorch:'How the Gaussian process ranked its candidates.',tpe:'See what TPE learned from and what it proposed.',overview:'Choose the parameters. Ask your own questions.',discover:'Rank parameters for the result you care about.',trials:'Find the signal in every experiment.',compare:'Understand what changed, and what worked.'}[state.view];
  $('#trial-count').textContent = data?.trials.length || 0;
  $('#compare-count').textContent = state.selected.length;
  document.title = `${state.run || 'Tuning'} · Graphspec`;
  if (!data) {
    $('#content').innerHTML = `<section class="panel">${state.run ? '<div class="empty"><span class="loading-dot"></span><h2>Loading this run</h2><p>Reading the saved trial records.</p></div>' : empty('No tuning runs found', 'Point the dashboard at a run directory or a folder containing tuning runs. Refresh after the first run is created.', '<button class="button" data-action="retry">Refresh runs</button>')}</section>`;
    return;
  }
  $('#content').innerHTML = state.view === 'current' ? currentRunView(data) : state.view === 'forest' ? forestView(data,state.proposal) : state.view === 'botorch' ? botorchView(data,state.proposal) : state.view === 'tpe' ? tpeView(data,state.proposal) : state.view === 'overview' ? overview() : state.view === 'discover' ? discoveryView(data,state,availableFields,filterDraft,discovery) : state.view === 'trials' ? trialsView() : compareView();
  for (const id of disclosures) {
    const element = document.getElementById(id);
    if (element) element.open = true;
  }
  for (const picker of opened) {
    const element = document.getElementById(picker.id);
    if (!element) continue;
    element.open = true;
    $('input',element).value = picker.query;
    $('.picker-options',element).innerHTML = pickerOptions(availableFields,picker.id,picker.query,element.dataset.configOnly === 'true');
    $('.picker-list',element).scrollTop = picker.scroll;
  }
  if (focusId && document.getElementById(focusId)) {
    document.getElementById(focusId).focus({preventScroll:true});
    if (selection) document.getElementById(focusId).setSelectionRange(...selection);
  } else if (focusKey) {
    const attribute = focusKey.replace(/[A-Z]/g,letter=>`-${letter.toLowerCase()}`);
    [...document.querySelectorAll(`[data-${attribute}]`)].find(element => element.dataset[focusKey] === focusValue)?.focus({preventScroll:true});
  }
  if (tableScroll && $('.table-wrap')) $('.table-wrap').scrollLeft = tableScroll;
  saveLocation();
}

function selectTrial(id, checked) {
  if (checked && !state.selected.includes(id)) {
    if (state.selected.length === 4) { toast('Compare up to four trials. Remove one to add another.'); return false; }
    state.selected.push(id);
  } else if (!checked) state.selected = state.selected.filter(value => value !== id);
  saveLocation();
  $('#compare-count').textContent = state.selected.length;
  if (state.view === 'trials') {
    const existing = $('.selection-bar');
    if (existing) existing.remove();
    if (state.selected.length) $('#content').insertAdjacentHTML('afterbegin',selectionBar());
  }
  return true;
}

async function loadDetail(id, updating = false) {
  if (detailRequest) detailRequest.abort();
  const controller = new AbortController();
  detailRequest = controller;
  detailId = id;
  if (!updating) {
    detailData = null;
    detailTab = 'summary';
    const trial = data.trials.find(value => value.id === id);
    $('#detail-title').textContent = `Trial ${String(trial.number).padStart(3,'0')}`;
    $('#detail-body').innerHTML = '<div class="empty"><span class="loading-dot"></span><p>Loading trial details…</p></div>';
    if (!$('#detail').open) $('#detail').showModal();
  }
  try {
    const result = await api('/api/trial?' + new URLSearchParams({run:state.run,id}),controller.signal);
    if (controller !== detailRequest || !$('#detail').open) return;
    const changed = JSON.stringify(detailData) !== JSON.stringify(result);
    detailData = result;
    if (!updating || changed) renderDetail();
  } catch (error) {
    if (error.name === 'AbortError' || controller !== detailRequest) return;
    if (updating) toast('Trial detail update failed. Displaying the previous details.');
    else $('#detail-body').innerHTML = `<div class="detail-content"><div class="notice">${esc(error.message)}</div><button class="button" data-action="retry-detail">Retry details</button></div>`;
  }
}

function renderDetail() {
  if (!detailData) return;
  const disclosures = [...$('#detail-body').querySelectorAll('details[id][open]')].map(element => element.id);
  const scroll = $('#detail').scrollTop;
  const focusId = $('#detail').contains(document.activeElement) ? document.activeElement.id : null;
  const logScroll = $('.detail-content pre')?.scrollTop || 0;
  const trial = {...data.trials.find(value => value.id === detailId), selector:detailData.selection.phase};
  const result = detailData.outcome;
  let body = '';
  if (detailTab === 'config') {
    body = `<div class="compare-heading"><h3>Full trial configuration</h3><button class="button" data-action="export-trial">↓ Export JSON</button></div><pre>${esc(JSON.stringify(detailData.config,null,2))}</pre><details id="proposal-record"><summary>Proposal record and search bounds</summary><pre>${esc(JSON.stringify(detailData.selection,null,2))}</pre></details>`;
  } else if (detailTab === 'logs') {
    body = detailData.log ? `<p class="small muted">${esc(detailData.log.name)}${detailData.log.truncated ? ' · showing the last 24 KB' : ''}</p><pre tabindex="0" aria-label="Worker log">${esc(detailData.log.text || 'The worker log is empty.')}</pre>` : '<p class="muted small">No worker log has been written for this trial.</p>';
    if (result?.traceback) body += `<h3>Failure traceback</h3><pre>${esc(result.traceback)}</pre>`;
  } else {
    const timings = result?.fit_phases || detailData.progress?.phase_seconds || {};
    const max = Math.max(1,...Object.values(timings).filter(finite));
    body = `<div class="run-meta">${badge(detailData.status)}<span>${esc(detailData.id)}</span></div>${result?.error ? `<div class="notice">${esc(result.error)}</div>` : ''}<dl class="result-lines">${fields({...Object.fromEntries(MEASURES.map(measure => [METRICS[measure.key],score(valueOf(trial,measure.key))])),'Trial duration':duration(trial.seconds)})}</dl><dl class="settings-grid">${fields({'Chosen by':SELECTORS[trial.selector] || label(trial.selector),'Time limit':duration(detailData.config.max_seconds),'Parameters':number(trial.parameters),'Epochs':result?.epochs === undefined ? `${detailData.progress?.epoch_seconds?.length || 0} / ${trial.config.epochs}` : `${result.epochs} / ${result.requested_epochs}`,'Optimizer updates':number(result?.optimizer_updates),'Peak GPU memory':finite(result?.peak_cuda_bytes) ? (result.peak_cuda_bytes/1073741824).toFixed(2)+' GiB' : 'Not recorded','Training rows':number(result?.train_unique_rows),'Holdout rows':number(result?.holdout_rows)})}</dl>${Object.keys(timings).length ? `<hr class="subtle-divider"><h3>Where the time went</h3>${Object.entries(timings).filter(([,value]) => finite(value)).map(([name,value]) => `<div class="timing-row"><span>${esc(label(name))}</span><div class="family-bar"><span style="width:${value/max*100}%"></span></div><span>${duration(value)}</span></div>`).join('')}` : '<p class="small muted">Phase timings have not been recorded yet.</p>'}${result?.epoch_losses?.length ? `<hr class="subtle-divider"><h3>Training loss by epoch</h3><div class="loss-list">${result.epoch_losses.map((loss,i) => `<span>Epoch ${i+1} · ${score(loss)}</span>`).join('')}</div><p class="small muted section-gap">Training losses are separate from the final holdout metrics.</p>` : ''}<details id="trial-record"><summary>Saved trial record</summary><pre>${esc(JSON.stringify(detailData.record,null,2))}</pre></details>`;
  }
  $('#detail-body').innerHTML = `<div class="detail-tabs" role="tablist" aria-label="Trial information">${['summary','config','logs'].map(tab => `<button id="detail-tab-${tab}" role="tab" aria-selected="${detailTab === tab}" aria-controls="detail-panel" tabindex="${detailTab === tab ? 0 : -1}" data-detail-tab="${tab}">${{summary:'Summary',config:'Configuration',logs:'Worker log'}[tab]}</button>`).join('')}</div><div id="detail-panel" class="detail-content" role="tabpanel" aria-labelledby="detail-tab-${detailTab}" tabindex="0">${body}</div>`;
  $('#detail').scrollTop = scroll;
  for (const id of disclosures) {
    const element = document.getElementById(id);
    if (element) element.open = true;
  }
  if (focusId) document.getElementById(focusId)?.focus({preventScroll:true});
  if ($('.detail-content pre')) $('.detail-content pre').scrollTop = logScroll;
}

function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content],{type}));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url),1000);
  toast(`Exported ${name}`);
}

function exportCsv() {
  const keys = ['number','id','status',...state.columns,'selector','mse','r2','seconds','parameters','config'];
  download(`${state.run}-trials.csv`,[keys.map(csvCell).join(','),...filtered().map(trial => keys.map(key => csvCell(/^(config|result)\//.test(key) || isMeasure(key) ? valueOf(trial,key) : trial[key])).join(','))].join('\r\n'),'text/csv;charset=utf-8');
}

function csvCell(value) {
  let text = value === null || value === undefined ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value);
  if (typeof value === 'string' && /^[\s]*[=+@-]/.test(text)) text = "'"+text;
  return '"'+text.replaceAll('"','""')+'"';
}

function exportGroups() {
  const groups = groupedResults(filterRows(data.trials,state),state.x,state.metric);
  const rows = [[fieldName(state.x),'trials','with_measure',higherIsBetter(state.metric) ? 'maximum' : 'minimum','median'],...groups.map(group => [displayValue(group.value),group.count,group.scored,group.best,group.median])];
  download(`${state.run}-grouped.csv`,rows.map(row => row.map(csvCell).join(',')).join('\r\n'),'text/csv;charset=utf-8');
}

function updateResults() {
  syncDiscovery();
  if (state.view === 'overview') $('#analysis-results').innerHTML = analysisResults(filterRows(data.trials,state),state);
  if (state.view === 'discover') {
    $('#discovery-results').innerHTML = discoveryResults(discovery,state.target);
    const button = $('[data-action="rank-parameters"]');
    button.disabled = discovery.busy || !state.target;
    button.textContent = discovery.busy ? 'Ranking…' : discovery.result ? 'Update ranking' : 'Rank parameters';
  }
  if (state.view === 'trials') {
    $('#trial-results').innerHTML = trialResults();
    $('#export').disabled = !filtered().length;
  }
  saveLocation();
}

document.addEventListener('click',event => {
  document.querySelectorAll('.field-picker[open]').forEach(picker => { if (!picker.contains(event.target)) picker.open = false; });
  const button = event.target.closest('button,[data-trial]');
  if (!button || button.disabled) return;
  if (button.dataset.view) { navigate(button.dataset.view); return; }
  if (button.dataset.trial && data) { loadDetail(button.dataset.trial); return; }
  if (button.dataset.pickerPath !== undefined) {
    const picker = button.closest('.field-picker');
    const current = pickerState(picker.id);
    current.path = JSON.parse(button.dataset.pickerPath);
    current.limit = 80;
    refreshPicker(picker);
    $('input',picker).focus({preventScroll:true});
    return;
  }
  if (button.dataset.pickerMore) {
    const picker = button.closest('.field-picker');
    const previous = $('.picker-list',picker).querySelectorAll('button').length;
    pickerState(picker.id).limit += 80;
    refreshPicker(picker);
    $('.picker-list',picker).querySelectorAll('button')[previous]?.focus();
    return;
  }
  if (button.dataset.discoverField) {
    const field = button.dataset.discoverField;
    if (!state.columns.includes(field)) state.columns.push(field);
    if (discovery.result && !discovery.result.categorical) {
      state.x = field;
      state.metric = state.target;
      navigate('overview');
    } else {
      if (/^(config|result)\//.test(state.target) && !state.columns.includes(state.target)) state.columns.push(state.target);
      state.sort = field;
      state.page = 1;
      navigate('trials');
    }
    return;
  }
  if (button.dataset.fieldChoice) {
    const field = button.dataset.fieldChoice;
    button.closest('details').open = false;
    const target = button.dataset.pickerTarget;
    if (target === 'target-field') state.target = field;
    if (target === 'axis-field') { state.x = field; if (/^(config|result)\//.test(field) && !state.columns.includes(field)) state.columns.push(field); }
    if (target === 'column-field' && !state.columns.includes(field)) state.columns.push(field);
    if (target === 'sort-field') { state.sort = field; state.page = 1; if (/^(config|result)\//.test(field) && !state.columns.includes(field)) state.columns.push(field); }
    if (target === 'filter-field') filterDraft = {...filterDraft,field,value:''};
    render();
    document.querySelector(`#${target} summary`)?.focus({preventScroll:true});
    return;
  }
  if (button.dataset.removeCondition !== undefined) { state.conditions.splice(Number(button.dataset.removeCondition),1); state.page = 1; render(); }
  if (button.dataset.removeColumn !== undefined) { state.columns.splice(Number(button.dataset.removeColumn),1); render(); }
  if (button.dataset.groupValue !== undefined) {
    const missing = button.dataset.groupValue === 'missing';
    state.conditions.push({field:state.x,op:missing ? 'missing' : 'is',value:missing ? '' : button.dataset.groupValue});
    state.page = 1;
    navigate('trials');
  }
  if (button.id === 'sort-direction') { state.direction = state.direction === 'asc' ? 'desc' : 'asc'; state.page = 1; render(); }
  if (button.dataset.sort) {
    if (state.sort === button.dataset.sort) state.direction = state.direction === 'asc' ? 'desc' : 'asc';
    else { state.sort = button.dataset.sort; state.direction = higherIsBetter(state.sort) ? 'desc' : 'asc'; }
    state.page = 1;
    render();
    [...document.querySelectorAll('[data-sort]')].find(element => element.dataset.sort === state.sort)?.focus({preventScroll:true});
  }
  if (button.dataset.page) { state.page += Number(button.dataset.page); render(); }
  if (button.dataset.remove) { selectTrial(button.dataset.remove,false); render(); }
  if (button.dataset.detailTab) { detailTab = button.dataset.detailTab; renderDetail(); $(`#detail-tab-${detailTab}`).focus(); }
  if (button.id === 'export') exportCsv();
  const action = button.dataset.action;
  if (action === 'rank-parameters') rankParameters();
  if (action === 'more-rankings') { discovery.limit += 30; $('#discovery-results').innerHTML = discoveryResults(discovery,state.target); }
  if (action === 'inspect-rows') navigate('trials');
  if (action === 'export-groups') exportGroups();
  if (action === 'clear-filters') { state.query = ''; state.status = 'all'; state.conditions = []; filterDraft = {field:'',op:'eq',value:''}; state.page = 1; render(); $('#search')?.focus(); }
  if (action === 'clear-selection') { state.selected = []; render(); }
  if (action === 'retry') refresh(true);
  if (action === 'retry-detail') loadDetail(detailId);
  if (action === 'export-trial') download(`${detailId}-config.json`,JSON.stringify(detailData.config,null,2),'application/json');
  if (action === 'export-configs') download(`${state.run}-comparison.json`,JSON.stringify(state.selected.map(id => { const trial = data.trials.find(t => t.id === id); return {trial:id,config:trial.config}; }),null,2),'application/json');
});

document.addEventListener('change',event => {
  const target = event.target;
  if (target.id === 'run') switchRun(target.value);
  if (target.id === 'proposal') { state.proposal = target.value; saveLocation(true); render(); }
  if (target.id === 'live') { schedule(); toast(target.checked ? 'Auto-refresh enabled, every 5 seconds.' : 'Auto-refresh paused. The tuning run continues.'); }
  if (target.id === 'status') { state[target.id] = target.value; state.page = 1; render(); }
  if (target.id === 'chart-metric') { state.metric = target.value; render(); }
  if (target.id === 'ranking-positive') { discovery.positive = target.checked; discovery.limit = 30; $('#discovery-results').innerHTML = discoveryResults(discovery,state.target); $('#ranking-positive').focus({preventScroll:true}); }
  if (target.id === 'filter-op') { filterDraft.op = target.value; render(); }
  if (target.id === 'differences') { state.differences = target.checked; render(); }
  if (target.dataset.select && !selectTrial(target.dataset.select,target.checked)) target.checked = false;
});

document.addEventListener('input',event => {
  if (event.target.dataset.pickerSearch) {
    const picker = event.target.closest('details');
    const current = pickerState(picker.id);
    current.query = event.target.value;
    current.limit = 80;
    refreshPicker(picker);
    return;
  }
  if (event.target.id === 'ranking-search') {
    const input = event.target, caret = input.selectionStart;
    discovery.query = input.value;
    discovery.limit = 30;
    $('#discovery-results').innerHTML = discoveryResults(discovery,state.target);
    $('#ranking-search').focus({preventScroll:true});
    $('#ranking-search').setSelectionRange(caret,caret);
    return;
  }
  if (event.target.id === 'filter-value') { filterDraft.value = event.target.value; return; }
  if (event.target.id !== 'search') return;
  state.query = event.target.value;
  state.page = 1;
  updateResults();
});

document.addEventListener('submit',event => {
  if (event.target.id !== 'condition-form') return;
  event.preventDefault();
  if (!filterDraft.field) return;
  if (['lt','lte','gt','gte'].includes(filterDraft.op) && (filterDraft.value.trim() === '' || !Number.isFinite(Number(filterDraft.value)))) {
    toast('Enter a finite number for a numeric condition.');
    $('#filter-value').focus();
    return;
  }
  state.conditions.push({...filterDraft});
  state.page = 1;
  render();
});

document.addEventListener('keydown',event => {
  if (event.target.dataset.pickerSearch && ['ArrowDown','Enter'].includes(event.key)) {
    event.preventDefault();
    const choices = event.target.closest('details').querySelectorAll('.picker-list button');
    if (event.key === 'Enter' && choices.length === 1) choices[0].click();
    else choices[0]?.focus();
  }
  if (event.target.closest('.picker-list') && ['ArrowDown','ArrowUp','Home','End'].includes(event.key)) {
    event.preventDefault();
    const buttons = [...event.target.closest('.picker-list').querySelectorAll('button')];
    const index = buttons.indexOf(event.target);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length-1 : Math.max(0,Math.min(buttons.length-1,index+(event.key === 'ArrowDown' ? 1 : -1)));
    buttons[next]?.focus();
  }
  if (event.target.closest('.field-picker') && event.key === 'Escape') {
    const picker = event.target.closest('details');
    picker.open = false;
    $('summary',picker).focus();
  }
  if (event.target.matches('circle[data-trial]') && ['Enter',' '].includes(event.key)) { event.preventDefault(); loadDetail(event.target.dataset.trial); }
  if (event.target.matches('circle[data-trial]') && ['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) {
    event.preventDefault();
    const points = [...document.querySelectorAll('circle[data-trial]')];
    const index = points.indexOf(event.target);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? points.length-1 : Math.max(0,Math.min(points.length-1,index+(event.key === 'ArrowRight' ? 1 : -1)));
    points.forEach((point,i) => point.setAttribute('tabindex',i === next ? '0' : '-1'));
    points[next].focus();
  }
  if (event.target.matches('[data-detail-tab]') && ['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) {
    event.preventDefault();
    const tabs = ['summary','config','logs'];
    const index = tabs.indexOf(detailTab);
    detailTab = tabs[event.key === 'Home' ? 0 : event.key === 'End' ? 2 : (index+(event.key === 'ArrowRight' ? 1 : 2))%3];
    renderDetail();
    $(`#detail-tab-${detailTab}`).focus();
  }
});

$('#refresh').addEventListener('click',() => refresh(true));
$('.skip').addEventListener('click',event => { event.preventDefault(); $('#main').focus(); $('#main').scrollIntoView({block:'start'}); });
$('.brand').addEventListener('click',event => { event.preventDefault(); navigate('overview'); });
$('#close-detail').addEventListener('click',() => $('#detail').close());
$('#detail').addEventListener('close',() => { detailRequest?.abort(); detailId = null; detailData = null; });
document.addEventListener('visibilitychange',() => { if (document.hidden) clearTimeout(poll); else if ($('#live').checked) refresh(true); });
window.addEventListener('popstate',() => {
  const previous = state.run;
  readLocation();
  if (previous !== state.run) { data = null; $('#detail').close(); render(); refresh(true); }
  else { render(); if (OPTIMIZER_VIEWS.includes(state.view) && data && !data.optimizer) refresh(); }
});

readLocation();
refresh(true);
