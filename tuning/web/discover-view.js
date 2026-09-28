import {fieldName} from './parameters.js';
import {filterToolbar,fieldPicker} from './explore.js';

const esc = value => String(value ?? '').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const percent = value => `${(value*100).toFixed(1)}%`;

export function discoveryView(data,state,fields,draft,discovery) {
  const count = state.conditions.length+(state.query ? 1 : 0)+(state.status !== 'all' ? 1 : 0);
  return `<section class="panel discovery-panel"><div class="analysis-controls">${fieldPicker(fields,'target-field',state.target,'Rank parameters for')}<button class="button primary" data-action="rank-parameters"${discovery.busy || !state.target ? ' disabled' : ''}>${discovery.busy ? 'Ranking…' : discovery.result ? 'Update ranking' : 'Rank parameters'}</button></div><details id="discovery-filters" class="discovery-filters"><summary>Trial filters${count ? ` · ${count} active` : ''}</summary>${filterToolbar(data,state,fields,draft)}</details><div id="discovery-results">${discoveryResults(discovery,state.target)}</div></section>`;
}

export function discoveryResults(discovery,target) {
  if (discovery.busy) return '<div class="analysis-empty" role="status"><span class="loading-dot"></span><p>Ranking parameters…</p></div>';
  if (discovery.error) return `<div class="notice error" role="alert">${esc(discovery.error)} Use Rank parameters to retry.</div>`;
  if (!discovery.result) return '<div class="analysis-empty"><h2>Find parameters worth exploring</h2><p>Choose a field, then rank the saved parameters.</p></div>';
  const {result} = discovery;
  const query = discovery.query.trim().toLowerCase();
  const rows = result.rankings.filter(row=>(!discovery.positive || row.gain>0) && row.fields.some(field=>fieldName(field).toLowerCase().includes(query)));
  const shown = rows.slice(0,discovery.limit);
  const totalRanked = result.rankings.reduce((sum,row)=>sum+row.fields.length,0);
  return `<div class="ranking-toolbar"><input type="search" id="ranking-search" aria-label="Find ranked parameters" placeholder="Find ranked parameters…" value="${esc(discovery.query)}"><label class="live-switch small"><input type="checkbox" id="ranking-positive"${discovery.positive ? ' checked' : ''}> Positive gain only</label><span class="small muted">Snapshot ${esc(discovery.at)}</span></div><div class="table-wrap"><table class="ranking-table" aria-label="Parameter ranking"><thead><tr><th scope="col">Parameter</th><th scope="col" title="Reduction in squared prediction error on held-out trials, relative to predicting their training-fold mean.">Predictive gain ↓</th><th scope="col">Explore</th></tr></thead><tbody>${shown.map(row=>{
    const field = row.fields.find(field=>fieldName(field).toLowerCase().includes(query));
    const aliases = row.fields.filter(value=>value!==field);
    return `<tr><td class="ranking-name"><button class="text-button" data-discover-field="${esc(field)}">${esc(fieldName(field))}</button>${aliases.length ? `<details id="rank-alias-${encodeURIComponent(row.fields[0])}"><summary>${aliases.length} fields with identical values</summary>${aliases.map(alias=>`<button class="text-button" data-discover-field="${esc(alias)}">${esc(fieldName(alias))}</button>`).join('')}</details>` : ''}</td><td><div class="ranking-gain"><span class="gain-track" aria-hidden="true"><span style="width:${Math.min(100,Math.max(0,row.gain*100))}%"></span></span><strong>${percent(row.gain)}</strong></div></td><td><button class="button" data-discover-field="${esc(field)}">${result.categorical ? 'Trials' : 'Graph'} →</button></td></tr>`;
  }).join('') || `<tr><td colspan="3" class="ranking-empty">${!result.rankings.length ? 'Not enough varying data to rank parameters.' : discovery.positive && !query ? 'No parameters improved predictions. Turn off Positive gain only to see all rankings.' : 'No matching parameters.'}</td></tr>`}</tbody></table></div><div class="ranking-footer"><span>${shown.length} of ${rows.length} ranked groups · ${totalRanked} parameters</span>${rows.length>discovery.limit ? '<button class="button" data-action="more-rankings">Show more</button>' : ''}</div>${Object.keys(result.skipped).length ? `<details class="ranking-skipped"><summary>Unranked parameters</summary>${Object.entries(result.skipped).map(([reason,count])=>`<p>${esc(reason)} <strong>${count}</strong></p>`).join('')}</details>` : ''}`;
}
