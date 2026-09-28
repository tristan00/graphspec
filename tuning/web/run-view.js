const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const finite = value => typeof value === 'number' && Number.isFinite(value);
const score = value => finite(value) ? value.toLocaleString('en-US', {maximumSignificantDigits:5}) : '—';
const count = value => finite(value) ? value.toLocaleString('en-US') : '—';
const duration = value => !finite(value) ? '—' : value < 60 ? `${value.toFixed(1)}s` : value < 3600 ? `${(value / 60).toFixed(1)}m` : `${(value / 3600).toFixed(1)}h`;
const percent = value => finite(value) ? `${(value * 100).toFixed(1)}%` : '—';
const words = value => String(value ?? 'unknown').replaceAll('_', ' ');
const badge = value => `<span class="badge ${esc(value)}">${esc(words(value))}</span>`;
const trialLink = trial => `<button class="trial-link" data-trial="${esc(trial.id)}">Trial ${trial.number}</button>`;
const message = text => `<p class="run-empty">${esc(text)}</p>`;
const facts = entries => `<dl class="run-facts">${entries.map(([name,value]) => `<div><dt>${esc(name)}</dt><dd>${esc(value)}</dd></div>`).join('')}</dl>`;
const surrogateName = selector => ({botorch:'BoTorch',forest:'SMAC'})[selector] || selector;
const mixText = mix => Object.entries(mix || {}).map(([name,share]) => `${surrogateName(name)} ${score(share)}%`).join(' · ') || '—';
const fixedParams = fixed => Object.keys(fixed || {}).length ? `<div class="fixed-params">${Object.entries(fixed).map(([key,value]) => `<span><span>${esc(key)}</span><strong>${esc(value)}</strong></span>`).join('')}</div>` : '<p class="run-muted">No parameters held random for this proposal.</p>';

export function activeSelection(data) {
  const progress = data.selection_progress;
  const started = Date.parse(data.runner?.started) / 1000;
  return data.runner?.status === 'running' && progress && progress.started >= started && progress.trial_number > data.trials.length ? progress : null;
}

export function runSummary(data) {
  const counts = {};
  for (const trial of data.trials) counts[trial.status] = (counts[trial.status] || 0) + 1;
  const complete = data.trials.filter(trial => trial.status === 'complete' && finite(trial.mse));
  return {counts, complete, best:complete.reduce((best,trial) => !best || trial.mse < best.mse ? trial : best, null), active:data.trials.find(trial => trial.status === 'running'), selection:activeSelection(data)};
}

function bestChart(trials) {
  const complete = trials.filter(trial => trial.status === 'complete' && finite(trial.mse));
  if (!complete.length) return message('The first completed trial will start the chart.');
  let best = Infinity;
  const points = complete.map(trial => { best = Math.min(best,trial.mse); return {...trial,best}; });
  const low = Math.min(...points.map(p => p.best));
  const high = Math.max(...points.map(p => p.best));
  const spread = high - low || Math.max(1,Math.abs(high) * .1);
  const x = trial => 70 + (trial.number - 1) / Math.max(1,trials.length - 1) * 850;
  const y = value => 205 - (value - low) / spread * 155;
  let previous = null;
  const path = points.map(point => { const command = previous ? `H${x(point)}V${y(point.best)}` : `M${x(point)},${y(point.best)}`; previous = point; return command; }).join(' ');
  const ticks = [0,.5,1].map(f => { const value = low + f * spread; return `<line x1="70" x2="920" y1="${y(value)}" y2="${y(value)}" class="run-grid"/><text x="60" y="${y(value)+4}" text-anchor="end">${score(value)}</text>`; }).join('');
  const improvements = points.filter((point,index) => !index || point.best < points[index-1].best);
  return `<div class="run-chart"><svg viewBox="0 0 950 260" role="img" aria-label="Best MSE by trial, lower is better">${ticks}<path d="${path}" class="run-curve"/>${improvements.map((trial,index) => `<circle class="point" cx="${x(trial)}" cy="${y(trial.best)}" r="5" data-trial="${esc(trial.id)}" tabindex="${index ? -1 : 0}" role="button" aria-label="Trial ${trial.number}, best MSE ${score(trial.best)}"><title>Trial ${trial.number} · ${score(trial.best)}</title></circle>`).join('')}<text x="70" y="237">Trial 1</text><text x="920" y="237" text-anchor="end">Trial ${trials.length}</text></svg></div>`;
}

function phaseTimings(progress) {
  const phases = Object.entries(progress?.phase_seconds || {}).filter(([,seconds]) => finite(seconds) && seconds > .01);
  const largest = Math.max(1,...phases.map(([,seconds]) => seconds));
  return phases.length ? `<div class="phase-timings">${phases.map(([phase,seconds]) => `<div><span>${esc(words(phase))}</span><span class="phase-track"><i style="width:${seconds/largest*100}%"></i></span><strong>${duration(seconds)}</strong></div>`).join('')}</div>` : '';
}

export function currentRunView(data) {
  const {counts,best,active,selection} = runSummary(data);
  const runner = data.runner;
  const status = runner?.status || 'unknown';
  const isRunning = status === 'running';
  const phase = active ? words(active.progress?.phase || 'Starting worker') : selection ? selection.stage : isRunning ? 'Between trials' : words(status);
  const elapsed = active?.started ? Math.max(0,data.recorded_at-Date.parse(active.started)/1000) : null;
  const budget = active ? active.config.max_seconds : data.plan.seconds;
  const budgetBound = data.plan.bounds?.max_seconds ?? data.plan.seconds;
  const newBudget = typeof budgetBound === 'object' ? `${duration(budgetBound.low)}–${duration(budgetBound.high)}` : duration(budgetBound);
  const timing = active?.progress;
  const remaining = finite(timing?.deadline) && finite(timing?.updated) && finite(timing?.recorded_at) ? Math.max(0,timing.deadline-timing.updated-Math.max(0,data.recorded_at-timing.recorded_at)) : null;
  const ratio = finite(remaining) && finite(budget) && budget > 0 ? Math.max(0,Math.min(100,(budget-remaining)/budget*100)) : null;
  const recent = data.trials.slice(-10).reverse();
  const epochs = active?.progress?.epoch_seconds;
  const last = data.trials.at(-1);
  const heartbeat = active?.progress?.recorded_at;
  return `<div class="run-stack">
    <section class="panel run-live"><div class="run-live-heading"><div><div class="eyebrow">${isRunning ? 'NOW' : 'RUN STATUS'}</div><h2>${esc(phase)}</h2></div>${badge(status)}</div>
      ${runner?.error ? `<div class="notice error run-error">${esc(runner.error)}</div>` : ''}
      ${data.stop_requested ? `<div class="notice">${isRunning ? 'Stop requested. The runner will exit after its current trial.' : 'Run stopped by request.'}</div>` : ''}
      <div class="run-now">${active ? `${trialLink(active)}<span>${esc(active.description)}</span>` : selection ? `<strong>Choosing trial ${selection.trial_number}</strong>${fixedParams(selection.fixed_params)}` : last ? `<span>Latest trial</span>${trialLink(last)}${badge(last.status)}` : '<span>No trials recorded yet.</span>'}</div>
      ${active ? `<div class="run-budget"><span>${finite(remaining) ? `${duration(remaining)} remaining` : `${duration(elapsed)} worker elapsed`}</span><span>${duration(budget)} trial budget</span></div>${ratio === null ? '' : `<progress aria-label="Trial budget used" max="100" value="${ratio}"></progress>`}<div class="run-inline">${Array.isArray(epochs) ? `<span>${epochs.length} / ${count(active.config.epochs)} epochs completed</span>` : ''}${finite(heartbeat) ? `<span>Worker update ${duration(Math.max(0,data.recorded_at-heartbeat))} ago</span>` : '<span>Waiting for the first worker update</span>'}</div>${phaseTimings(active.progress)}` : ''}
      <div class="run-inline run-totals">${Object.entries(counts).map(([name,value]) => `<span><strong>${count(value)}</strong> ${esc(words(name))}</span>`).join('') || '<span>0 trials</span>'}</div>
    </section>
    <section class="panel"><div class="panel-head"><div><h2>Best result over time</h2><p>MSE · lower is better</p></div>${best ? `<div class="run-best"><strong>${score(best.mse)}</strong>${trialLink(best)}</div>` : ''}</div>${bestChart(data.trials)}</section>
    <section class="panel"><div class="panel-head"><h2>Recent trials</h2><button class="button quiet" data-view="trials">All trials →</button></div>${recent.length ? `<div class="table-wrap"><table class="run-table"><thead><tr><th>Trial</th><th>Status</th><th>Chosen by</th><th>MSE</th><th>Duration</th></tr></thead><tbody>${recent.map(trial => `<tr><td>${trialLink(trial)}</td><td>${badge(trial.status)}</td><td>${esc(surrogateName(trial.selector))}</td><td>${score(trial.mse)}</td><td>${duration(trial.seconds)}</td></tr>`).join('')}</tbody></table></div>` : message('No trials recorded yet.')}</section>
    <details class="panel run-settings" id="run-settings"><summary>Run settings</summary>${facts([['Optimizer',data.plan.selector],['Minimum history',count(data.plan.random_trials)],['Keep random',(data.plan.random_params || []).join(', ') || 'None'],['Selector mix',mixText(data.plan.mix)],['New trial budget',newBudget],['Workers',count(data.plan.workers)],['Training rows',count(data.plan.train_rows)],['Holdout rows',count(data.plan.holdout_rows)],['Trial limit',data.plan.limit === null ? 'Unlimited' : count(data.plan.limit)]])}${Object.keys(data.plan.bounds || {}).length ? `<div class="run-bounds"><h3>Parameter bounds</h3><pre>${esc(JSON.stringify(data.plan.bounds,null,2))}</pre></div>` : ''}</details>
  </div>`;
}

function histogramChart(title, hist, marks) {
  const {edges,counts} = hist;
  const low = edges[0], high = edges.at(-1), span = high-low || 1, peak = Math.max(1,...counts);
  const x = value => 40 + (value-low)/span * 400;
  const bars = counts.map((n,i) => `<rect x="${x(edges[i])}" y="${150-n/peak*120}" width="${Math.max(1,x(edges[i+1])-x(edges[i])-1)}" height="${n/peak*120}"/>`).join('');
  const lines = marks.filter(mark => finite(mark.value) && mark.value >= low && mark.value <= high).map(mark => `<line class="run-mark ${esc(mark.kind)}" x1="${x(mark.value)}" x2="${x(mark.value)}" y1="22" y2="150"/><text x="${x(mark.value)}" y="14" text-anchor="middle">${esc(mark.label)} ${score(mark.value)}</text>`).join('');
  return `<figure class="run-histogram"><figcaption>${esc(title)}</figcaption><svg viewBox="0 0 480 175" role="img" aria-label="${esc(title)} across all candidates">${bars}${lines}<line class="run-grid" x1="40" x2="440" y1="150" y2="150"/><text x="40" y="168">${score(low)}</text><text x="440" y="168" text-anchor="end">${score(high)}</text></svg></figure>`;
}

function distributionPanel(diagnostics) {
  const chosen = diagnostics.candidates[0], d = diagnostics.distribution;
  const charts = [
    ['Predicted MSE', d.predicted_mse, [{value:chosen.predicted_mse,label:'Chosen',kind:'chosen'},{value:diagnostics.best_observed_mse,label:'Best so far',kind:'best'}]],
    ['Uncertainty (±1σ)', d.uncertainty, [{value:chosen.uncertainty,label:'Chosen',kind:'chosen'}]],
    ['Chance of beating best', d.beat_probability, [{value:chosen.beat_probability,label:'Chosen',kind:'chosen'}]],
    ['Completion probability', d.completion_probability, [{value:chosen.completion_probability,label:'Chosen',kind:'chosen'}]],
  ].filter(([,hist]) => hist).map(([title,hist,marks]) => histogramChart(title,hist,marks));
  return `<section class="panel"><div class="panel-head"><div><h2>Where the ${count(diagnostics.candidates_drawn)} candidates landed</h2><p>Count of candidates per bin</p></div></div><div class="run-histograms">${charts.join('')}</div></section>`;
}

function importanceList(title, entries) {
  const largest = Math.max(1e-12,...entries.map(entry => entry.value));
  return `<div><h3>${esc(title)}</h3><div class="phase-timings importance">${entries.map(entry => `<div><span title="${esc(entry.name)}">${esc(entry.name)}</span><span class="phase-track"><i style="width:${entry.value/largest*100}%"></i></span><strong>${(entry.value*100).toFixed(1)}%</strong></div>`).join('')}</div></div>`;
}

function importancePanel(importance) {
  const lists = [importanceList('Score model', importance.score || [])];
  if (importance.completion) lists.push(importanceList('Completion model', importance.completion));
  return `<section class="panel"><div class="panel-head"><div><h2>What the forest split on</h2><p>Share of impurity reduction attributed to each parameter · summed over its encoded columns</p></div></div><div class="panel-body importance-grid">${lists.join('')}</div></section>`;
}

const signed = value => finite(value) ? `${value >= 0 ? '+' : '−'}${score(Math.abs(value))}` : '—';
const chance = value => !finite(value) ? '—' : value >= .001 ? percent(value) : value === 0 ? '0%' : `${(value*100).toPrecision(2)}%`;
const position = (rank, total) => finite(rank) ? `#${count(rank)} of ${count(total)}` : '—';

function whyPanel(proposal, trial, diagnostics) {
  const chosen = diagnostics.candidates[0], next = diagnostics.candidates[1], rank = chosen.rank || {}, total = diagnostics.candidates_drawn, best = diagnostics.best_observed_mse;
  const constrained = finite(chosen.completion_probability);
  const weight = proposal.settings?.completion_weight ?? 1;
  const completionTerm = candidate => weight * Math.log(candidate.completion_probability);
  const logCompletion = constrained ? completionTerm(chosen) : 0;
  const terms = [['log improvement',chosen.log_improvement,next?.log_improvement]];
  if (constrained) terms.push([weight === 1 ? 'log completion' : `${score(weight)} × log completion`,logCompletion,next ? completionTerm(next) : null]);
  const leads = terms.map(([name,value,other]) => [name,finite(other) ? value-other : null]);
  const driver = leads.filter(([,lead]) => finite(lead)).sort((a,b) => b[1]-a[1])[0];
  const beat = chosen.beat_probability;
  const story = finite(beat) ? beat < .05
    ? `No candidate is predicted to beat ${score(best)} outright; this one's chance is ${chance(beat)}. Expected improvement then comes from the tail of each prediction, which a lower predicted MSE or a wider uncertainty both extend${constrained ? ', discounted by completion probability' : ''}. This candidate is ${position(rank.predicted_mse,total)} on predicted MSE and ${position(rank.uncertainty,total)} on uncertainty.`
    : `Predicted to beat ${score(best)} with ${chance(beat)} chance${constrained ? `, and to complete with ${percent(chosen.completion_probability)}` : ''}.`
    : '';
  const rows = [
    ['Predicted MSE',score(chosen.predicted_mse)],['Predicted MSE rank (lowest first)',position(rank.predicted_mse,total)],
    ['Uncertainty (±1σ)',score(chosen.uncertainty)],['Uncertainty rank (widest first)',position(rank.uncertainty,total)],
  ];
  if (finite(beat)) rows.push([`Chance of beating ${score(best)}`,chance(beat)],['Chance rank',position(rank.beat_probability,total)]);
  if (constrained) rows.push(['Predicted completion',percent(chosen.completion_probability)],['Completion rank',position(rank.completion_probability,total)]);
  rows.push(['Log improvement',score(chosen.log_improvement)]);
  if (constrained) rows.push([weight === 1 ? 'Log completion' : `${score(weight)} × log completion`,score(logCompletion)]);
  rows.push(['Rank score',score(chosen.acquisition)]);
  if (next) rows.push(['Lead over runner-up',score(chosen.acquisition-next.acquisition)],...leads.map(([name,lead]) => [`Lead from ${name}`,signed(lead)]));
  rows.push(['Actual MSE',finite(trial?.mse) ? score(trial.mse) : trial && ['failed','timeout','interrupted'].includes(trial.status) ? words(trial.status) : 'Awaiting result']);
  return `<section class="panel"><div class="panel-head"><div><h2>Why this candidate</h2><p>${constrained ? `Rank score = log improvement + ${weight === 1 ? '' : `${score(weight)} × `}log completion` : 'Rank score = log improvement'} · ranks are among the ${count(total)} candidates drawn</p></div></div><div class="panel-body">${story ? `<p class="run-muted">${esc(story)}${driver ? ` Its lead over the runner-up comes from ${esc(driver[0])}.` : ''}</p>` : ''}${facts(rows)}</div></section>`;
}

function candidateTable(diagnostics, weight) {
  const candidates = diagnostics.candidates;
  const constrained = candidates.some(c => finite(c.completion_probability));
  const beats = candidates.some(c => finite(c.beat_probability));
  const ranked = candidates.some(c => c.rank);
  const metric = (value, rank) => `<td>${value}</td>${ranked ? `<td class="run-rank">${finite(rank) ? '#' + count(rank) : '—'}</td>` : ''}`;
  const header = (name, order) => `<th>${name}</th>${ranked ? `<th class="run-rank">rank<br><small>${order}</small></th>` : ''}`;
  return `<section class="panel"><div class="panel-head"><div><h2>Top ${count(candidates.length)} of ${count(diagnostics.candidates_drawn)} candidates</h2><p>Ordered by rank score, higher is better · each metric's rank is its position among all ${count(diagnostics.candidates_drawn)} candidates</p></div></div><div class="table-wrap"><table class="run-table"><thead><tr><th>Rank</th><th>Rank score</th><th>Log improvement</th>${constrained ? `<th>${weight === 1 ? 'Log completion' : `${score(weight)} × log completion`}</th>` : ''}${header('Predicted MSE','lowest first')}${header('Uncertainty (±1σ)','widest first')}${beats ? header('Beats best','highest first') : ''}${constrained ? header('Completion','highest first') : ''}<th>Configuration</th></tr></thead><tbody>${candidates.map((c,index) => { const r = c.rank || {}; return `<tr${index ? '' : ' class="run-winner"'}><td>${index ? index + 1 : 'Chosen'}</td><td>${score(c.acquisition)}</td><td>${score(c.log_improvement)}</td>${constrained ? `<td>${score(weight * Math.log(c.completion_probability))}</td>` : ''}${metric(score(c.predicted_mse),r.predicted_mse)}${metric(score(c.uncertainty),r.uncertainty)}${beats ? metric(chance(c.beat_probability),r.beat_probability) : ''}${constrained ? metric(percent(c.completion_probability),r.completion_probability) : ''}<td class="run-param-cell">${c.config ? `<details id="candidate-${index}" class="candidate-config"><summary>Configuration</summary><pre>${esc(JSON.stringify(c.config,null,2))}</pre></details>` : '—'}</td></tr>`; }).join('')}</tbody></table></div></section>`;
}

function surrogateView(data, selectedId, selector) {
  const {counts,selection} = runSummary(data);
  const finished = (counts.complete || 0) + (counts.failed || 0) + (counts.timeout || 0);
  const minimum = data.plan.random_trials;
  const status = data.runner?.status || 'unknown';
  const stage = selection ? selection.stage : status !== 'running' ? `Run ${words(status)}` : data.plan.selector === selector ? finished < minimum ? 'Gathering history' : `${surrogateName(selector)} active` : `Run optimizer: ${surrogateName(data.plan.selector)}`;
  const progress = `<section class="panel optimizer-status"><div class="run-inline"><strong>${esc(stage)}</strong><span>${count(finished)} finished trials · ${count(minimum)} minimum</span><span>Keep random: ${esc((data.plan.random_params || []).join(', ') || 'None')}</span><span>Mix: ${esc(mixText(data.plan.mix))}</span></div>${selection ? `<div class="run-now"><span>Choosing trial ${selection.trial_number}</span>${fixedParams(selection.fixed_params)}</div>` : ''}</section>`;
  if (!data.optimizer) return progress + message(`Loading ${surrogateName(selector)} proposals…`);
  const proposals = data.optimizer.proposals.filter(item => item.selector === selector);
  if (!proposals.length) return progress + `<section class="panel">${message(`No ${surrogateName(selector)} proposals yet.`)}</section>`;
  const proposal = selectedId ? proposals.find(item => item.id === selectedId) : proposals.at(-1);
  const picker = `<div class="proposal-picker"><label for="proposal">Proposal</label><select id="proposal"><option value=""${!selectedId ? ' selected' : ''}>Latest proposal</option>${selectedId && !proposal ? `<option value="${esc(selectedId)}" selected>Unavailable proposal</option>` : ''}${[...proposals].reverse().map(item => `<option value="${esc(item.id)}"${selectedId === item.id ? ' selected' : ''}>Trial ${item.number}</option>`).join('')}</select></div>`;
  if (!proposal) return `<div class="run-stack">${progress}<section class="panel"><div class="panel-head">${picker}</div>${message('This proposal is no longer available. Choose another proposal.')}</section></div>`;
  const trial = data.trials.find(item => item.id === proposal.id);
  const prediction = proposal.diagnostics?.prediction;
  const diagnostics = proposal.diagnostics?.candidates?.length ? {...proposal.diagnostics, candidates_drawn:proposal.settings?.candidates} : null;
  return `<div class="run-stack">${progress}<section class="panel"><div class="panel-head"><div><h2>${trial ? trialLink(trial) : `Trial ${proposal.number}`}</h2>${trial ? badge(trial.status) : ''}</div>${picker}</div><div class="panel-body"><h3>Random choices held fixed</h3>${fixedParams(proposal.fixed_params)}<div class="run-inline optimizer-training"><span>Learned from <strong>${count(proposal.observations)}</strong> completed + <strong>${count(proposal.failed_observations)}</strong> failed trials</span><span>${esc(surrogateName(proposal.selector))} · ${esc(proposal.settings?.model || '')}</span><span>Selection took ${duration(proposal.seconds)}</span></div>
      ${diagnostics ? '' : prediction ? facts([['Predicted MSE',score(prediction.predicted_mse)],['Uncertainty (±1σ)',score(prediction.uncertainty)],['Predicted completion',percent(prediction.completion_probability)],['MSE to beat',score(proposal.diagnostics?.best_observed_mse)],['Actual MSE',finite(trial?.mse) ? score(trial.mse) : trial && ['failed','timeout','interrupted'].includes(trial.status) ? words(trial.status) : 'Awaiting result']]) : message('Predictions were not recorded for this earlier proposal.')}
    </div></section>
    ${diagnostics ? whyPanel(proposal,trial,diagnostics) : ''}
    ${diagnostics?.importance ? importancePanel(diagnostics.importance) : ''}
    ${diagnostics?.distribution ? distributionPanel(diagnostics) : ''}
    ${diagnostics ? candidateTable(diagnostics,proposal.settings?.completion_weight ?? 1) : ''}
    <section class="panel"><div class="panel-head"><h2>Proposal outcomes</h2></div><div class="table-wrap"><table class="run-table"><thead><tr><th>Trial</th><th>Predicted MSE</th><th>Uncertainty (±1σ)</th><th>Actual MSE</th><th>Predicted completion</th><th>Status</th><th>Random choices</th></tr></thead><tbody>${[...proposals].reverse().map(item => { const outcome = data.trials.find(t => t.id === item.id); const p = item.diagnostics?.prediction; return `<tr><td>${outcome ? trialLink(outcome) : `Trial ${item.number}`}</td><td>${score(p?.predicted_mse)}</td><td>${score(p?.uncertainty)}</td><td>${score(outcome?.mse)}</td><td>${percent(p?.completion_probability)}</td><td>${outcome ? badge(outcome.status) : '—'}</td><td class="run-param-cell">${Object.entries(item.fixed_params || {}).map(([name,value]) => `${esc(name)} = ${esc(value)}`).join('<br>') || 'None'}</td></tr>`; }).join('')}</tbody></table></div></section></div>`;
}

const flatten = config => ({...Object.fromEntries(Object.entries(config.model || {}).map(([name,value]) => ['model/' + name,value])),...Object.fromEntries(Object.entries(config.graph || {}).map(([name,value]) => ['graph/' + name,value])),...Object.fromEntries(Object.entries(config).filter(([name]) => !['seed','model','graph'].includes(name)))});

function goodSetSummary(name, value, configs) {
  const values = configs.map(config => config[name]).filter(item => item !== undefined);
  if (!values.length) return 'not used';
  if (typeof value === 'number' && values.every(item => typeof item === 'number')) {
    const sorted = [...values].sort((a,b) => a-b);
    return `${score(sorted[0])} – ${score(sorted.at(-1))}, median ${score(sorted[Math.floor(sorted.length/2)])}`;
  }
  return `${count(values.filter(item => item === value).length)} of ${count(values.length)} share it`;
}

export function tpeView(data, selectedId) {
  const {counts,selection} = runSummary(data);
  const finished = (counts.complete || 0) + (counts.failed || 0) + (counts.timeout || 0);
  const status = data.runner?.status || 'unknown';
  const stage = selection ? selection.stage : status !== 'running' ? `Run ${words(status)}` : data.plan.selector === 'tpe' ? 'TPE active' : `Run optimizer: ${data.plan.selector}`;
  const progress = `<section class="panel optimizer-status"><div class="run-inline"><strong>${esc(stage)}</strong><span>${count(finished)} finished trials</span><span>Mix: ${esc(mixText(data.plan.mix))}</span></div></section>`;
  if (!data.optimizer) return progress + message('Loading TPE proposals…');
  const proposals = data.optimizer.proposals.filter(item => item.selector === 'tpe');
  if (!proposals.length) return progress + `<section class="panel">${message('No TPE proposals yet.')}</section>`;
  const proposal = selectedId ? proposals.find(item => item.id === selectedId) : proposals.at(-1);
  const picker = `<div class="proposal-picker"><label for="proposal">Proposal</label><select id="proposal"><option value=""${!selectedId ? ' selected' : ''}>Latest proposal</option>${selectedId && !proposal ? `<option value="${esc(selectedId)}" selected>Unavailable proposal</option>` : ''}${[...proposals].reverse().map(item => `<option value="${esc(item.id)}"${selectedId === item.id ? ' selected' : ''}>Trial ${item.number}</option>`).join('')}</select></div>`;
  if (!proposal) return `<div class="run-stack">${progress}<section class="panel"><div class="panel-head">${picker}</div>${message('This proposal is no longer available. Choose another proposal.')}</section></div>`;
  const trial = data.trials.find(item => item.id === proposal.id);
  const excluded = new Set((proposal.excluded || []).map(item => item.trial));
  const used = data.trials.filter(item => item.number < proposal.number && !excluded.has(item.id) && ['complete','failed','timeout'].includes(item.status));
  const good = used.filter(item => item.status === 'complete' && finite(item.mse)).sort((a,b) => a.mse-b.mse).slice(0,proposal.settings?.good_set ?? Math.min(25,Math.ceil(used.length*.1)));
  const goodConfigs = good.map(item => flatten(item.config || {}));
  const usedConfigs = used.map(item => flatten(item.config || {}));
  const params = Object.entries(proposal.params || {});
  return `<div class="run-stack">${progress}<section class="panel"><div class="panel-head"><div><h2>${trial ? trialLink(trial) : `Trial ${proposal.number}`}</h2>${trial ? badge(trial.status) : ''}</div>${picker}</div><div class="panel-body">
      <div class="run-inline optimizer-training"><span>Learned from <strong>${count(proposal.observations)}</strong> completed + <strong>${count(proposal.failed_observations)}</strong> failed trials</span><span><strong>${count((proposal.excluded || []).length)}</strong> trials excluded by the bounds</span><span>Selection took ${duration(proposal.seconds)}</span></div>
      <p class="run-muted">TPE splits the trials it can use into a good set, the best ${count(good.length)} by MSE, and the rest, with failed and timed-out trials always in the rest. It builds one density from each set, draws candidates from the good density, and keeps the candidate whose good-to-rest density ratio is highest: the one that looks most like a good trial and least like the others. It never draws far from the good set.</p>
      ${facts([['Actual MSE',finite(trial?.mse) ? score(trial.mse) : trial && ['failed','timeout','interrupted'].includes(trial.status) ? words(trial.status) : 'Awaiting result'],['Best in good set',score(good[0]?.mse)],['Worst in good set',score(good.at(-1)?.mse)]])}
      <h3>Good set</h3><div class="fixed-params">${good.map(item => `<span>${trialLink(item)}<strong>${score(item.mse)}</strong></span>`).join('') || '<p class="run-muted">No completed trials in range.</p>'}</div>
    </div></section>
    <details class="panel run-settings" id="tpe-params"><summary>Proposed parameters against the good set · ${count(params.length)} parameters</summary><div class="table-wrap"><table class="run-table"><thead><tr><th>Parameter</th><th>Proposed</th><th>Good set (${count(good.length)})</th><th>All used trials (${count(used.length)})</th></tr></thead><tbody>${params.map(([name,value]) => `<tr><td class="run-param-cell">${esc(name)}</td><td>${esc(value)}</td><td>${esc(goodSetSummary(name,value,goodConfigs))}</td><td>${esc(goodSetSummary(name,value,usedConfigs))}</td></tr>`).join('')}</tbody></table></div></details>
    <section class="panel"><div class="panel-head"><h2>Proposal outcomes</h2></div><div class="table-wrap"><table class="run-table"><thead><tr><th>Trial</th><th>Learned from</th><th>Excluded</th><th>Actual MSE</th><th>Status</th></tr></thead><tbody>${[...proposals].reverse().map(item => { const outcome = data.trials.find(t => t.id === item.id); return `<tr><td>${outcome ? trialLink(outcome) : `Trial ${item.number}`}</td><td>${count(item.observations)} + ${count(item.failed_observations)} failed</td><td>${count((item.excluded || []).length)}</td><td>${score(outcome?.mse)}</td><td>${outcome ? badge(outcome.status) : '—'}</td></tr>`; }).join('')}</tbody></table></div></section></div>`;
}

export function botorchView(data, selectedId) {
  return surrogateView(data, selectedId, 'botorch');
}

export function forestView(data, selectedId) {
  return surrogateView(data, selectedId, 'forest');
}
