import test from 'node:test';
import assert from 'node:assert/strict';
import {activeSelection,runSummary,currentRunView,botorchView,forestView,tpeView} from '../tuning/web/run-view.js';

const fixture = () => ({
  recorded_at:200, runner:{status:'running',started:new Date(100000).toISOString()},
  plan:{selector:'botorch',random_trials:2,random_params:['model/view'],seconds:120,workers:2,bounds:{},limit:null},
  trials:[{id:'a',number:1,status:'complete',mse:0,seconds:10,selector:'tpe',config:{}}, {id:'b',number:2,status:'failed',mse:null,selector:'tpe',config:{}}],
  optimizer:{proposals:[]},
});

test('current summary includes failed history and preserves a zero best score', () => {
  const data=fixture(), summary=runSummary(data), html=currentRunView(data);
  assert.equal(summary.best.id,'a');
  assert.deepEqual(summary.counts,{complete:1,failed:1});
  assert.match(html,/Best result over time/);
  assert.match(html,/Trial 1/);
  assert.doesNotMatch(html,/NaN|Infinity|undefined/);
  assert.match(botorchView(data,''),/2 finished trials/);
});

test('only selection from the live runner and an uncommitted trial appears live', () => {
  const data=fixture();
  data.selection_progress={started:101,trial_number:3,stage:'Fitting models'};
  assert.equal(activeSelection(data).stage,'Fitting models');
  data.selection_progress.started=99;
  assert.equal(activeSelection(data),null);
  data.selection_progress.started=101;
  data.selection_progress.trial_number=2;
  assert.equal(activeSelection(data),null);
  data.selection_progress.trial_number=3;
  data.runner.status='failed';
  assert.equal(activeSelection(data),null);
  assert.match(botorchView(data,''),/Run failed/);
  assert.doesNotMatch(botorchView(data,''),/active/);
});

test('running progress stays bounded and escapes trial text and runner errors', () => {
  const data=fixture();
  data.trials.push({id:'<script>',number:3,status:'running',description:'<img onerror=x>',started:new Date(0).toISOString(),config:{epochs:2,max_seconds:120},progress:{phase:'training',recorded_at:199,deadline:500,updated:600,epoch_seconds:[20],phase_seconds:{training:21}}});
  data.runner.error='<script>bad</script>';
  const html=currentRunView(data);
  assert.match(html,/value="100"/);
  assert.match(html,/1 \/ 2 epochs completed/);
  assert.match(html,/&lt;img onerror=x&gt;/);
  assert.doesNotMatch(html,/<script>|<img/);
  data.trials.at(-1).progress.deadline=661;
  data.plan.seconds=600;
  assert.match(currentRunView(data),/value="50"/);
});

test('empty runs and single completed scores render without invalid chart coordinates', () => {
  const data=fixture();
  for (const size of [0,1]) {
    data.trials=fixture().trials.slice(0,size);
    assert.doesNotMatch(currentRunView(data),/NaN|Infinity|undefined/);
  }
  assert.doesNotMatch(botorchView(data,''),/data-trial="e"/);
});

test('new trial budget shows its range while the active trial retains its exact limit', () => {
  const data=fixture();
  data.plan.bounds={epochs:{low:1,high:5},max_seconds:{low:600,high:1200}};
  data.plan.random_params=['model/view','epochs','max_seconds'];
  data.trials.push({id:'active',number:3,status:'running',config:{epochs:4,max_seconds:840},progress:{epoch_seconds:[20]}});
  const html=currentRunView(data);
  assert.match(html,/New trial budget<\/dt><dd>10.0m–20.0m/);
  assert.match(html,/14.0m trial budget/);
  assert.match(html,/1 \/ 4 epochs completed/);
  assert.match(html,/model\/view, epochs, max_seconds/);
});

test('proposal switching distinguishes historical diagnostics, results, and unavailable selections', () => {
  const data=fixture();
  const prediction={predicted_mse:1.2,uncertainty:.3,completion_probability:.8};
  data.optimizer.proposals=[{id:'a',number:1,selector:'botorch',observations:2,failed_observations:1,fixed_params:{'model/view':'snapshot'}}, {id:'b',number:2,selector:'botorch',observations:3,failed_observations:2,fixed_params:{'model/view':'<new>'},settings:{candidates:12,completion_weight:2},diagnostics:{prediction,best_observed_mse:1.5,distribution:{predicted_mse:{edges:[1,1.5,2],counts:[5,7]},acquisition:{edges:[-5,-3],counts:[12]},completion_probability:{edges:[0,1],counts:[12]}},candidates:[{...prediction,acquisition:-3,log_improvement:-2.8,beat_probability:.01,rank:{predicted_mse:1,uncertainty:1,completion_probability:6,beat_probability:1,acquisition:1},config:{model:{view:'walks','snapshot/layers':2},lr:.1,epochs:3}},{predicted_mse:1.4,uncertainty:.1,completion_probability:.5,acquisition:-4,log_improvement:-3.3,beat_probability:.002,config:{model:{view:'walks'},lr:.2,epochs:3}}]}}];
  assert.match(botorchView(data,''),/Top 2 of 12 candidates/);
  assert.doesNotMatch(botorchView(data,''),/id="candidates"/);
  assert.match(botorchView(data,''),/beating 1.5<\/dt><dd>1.0%/);
  assert.match(botorchView(data,''),/Lead over runner-up<\/dt><dd>1<\/dd>/);
  assert.match(botorchView(data,''),/Lead from 2 × log completion<\/dt><dd>\+0.94/);
  assert.match(botorchView(data,''),/Rank score = log improvement \+ 2 × log completion/);
  assert.match(botorchView(data,''),/Predicted MSE rank \(lowest first\)<\/dt><dd>#1 of 12/);
  assert.match(botorchView(data,''),/No candidate is predicted to beat 1.5 outright; this one&#39;s chance is 1.0%.*This candidate is #1 of 12 on predicted MSE and #1 of 12 on uncertainty/);
  assert.match(botorchView(data,''),/<td>0.2%<\/td>/);
  assert.match(botorchView(data,''),/<td>Chosen<\/td><td>-3<\/td>/);
  assert.match(botorchView(data,''),/<details id="candidate-0" class="candidate-config"><summary>Configuration<\/summary><pre>\{\n  &quot;model&quot;: \{\n    &quot;view&quot;: &quot;walks&quot;,\n    &quot;snapshot\/layers&quot;: 2\n  \},\n  &quot;lr&quot;: 0.1,\n  &quot;epochs&quot;: 3\n\}<\/pre><\/details>/);
  assert.match(botorchView(data,''),/Where the 12 candidates landed/);
  assert.match(botorchView(data,''),/Best so far 1.5/);
  assert.match(botorchView(data,''),/<details id="candidate-1" class="candidate-config"><summary>Configuration<\/summary><pre>\{\n  &quot;model&quot;: \{\n    &quot;view&quot;: &quot;walks&quot;\n  \},\n  &quot;lr&quot;: 0.2,\n  &quot;epochs&quot;: 3\n\}<\/pre><\/details>/);
  assert.match(botorchView(data,''),/Actual MSE<\/dt><dd>failed/);
  assert.match(botorchView(data,''),/Uncertainty rank \(widest first\)<\/dt><dd>#1 of 12/);
  assert.match(botorchView(data,''),/<td>1.2<\/td><td>0.3<\/td><td>—<\/td><td>80.0%<\/td>/);
  assert.match(botorchView(data,''),/<td>Chosen<\/td><td>-3<\/td><td>-2.8<\/td><td>-0.44629<\/td><td>1.2<\/td><td class="run-rank">#1<\/td><td>0.3<\/td><td class="run-rank">#1<\/td>/);
  assert.match(botorchView(data,''),/&lt;new&gt;/);
  assert.match(botorchView(data,'a'),/Predictions were not recorded for this earlier proposal/);
  assert.doesNotMatch(botorchView(data,'a'),/candidates<\/h2>/);
  assert.match(botorchView(data,'missing'),/Choose another proposal/);
});

test('tpe view shows the good set, parameter placement, and ignores botorch proposals', () => {
  const data=fixture();
  data.plan.selector='tpe';
  data.trials=[
    {id:'a',number:1,status:'complete',mse:1.0,selector:'random',config:{model:{view:'snapshot'},lr:.01,epochs:3}},
    {id:'b',number:2,status:'complete',mse:2.0,selector:'random',config:{model:{view:'walks'},lr:.02,epochs:3}},
    {id:'c',number:3,status:'failed',mse:null,selector:'random',config:{model:{view:'walks'},lr:.03,epochs:3}},
    {id:'d',number:4,status:'complete',mse:1.5,selector:'random',config:{model:{view:'snapshot'},lr:.04,epochs:5}},
    {id:'e',number:5,status:'running',mse:null,selector:'tpe',config:{model:{view:'snapshot'},lr:.015,epochs:4}},
  ];
  data.optimizer.proposals=[{id:'z',number:0,selector:'botorch'},{id:'e',number:5,selector:'tpe',observations:2,failed_observations:1,seconds:3,excluded:[{trial:'d',status:'complete',parameters:['epochs']}],params:{'model/view':'snapshot',lr:.015,epochs:4}}];
  const html=tpeView(data,'');
  assert.match(html,/1<\/strong> trials excluded by the bounds/);
  assert.match(html,/the best 1 by MSE/);
  data.optimizer.proposals[1].settings={good_set:2};
  assert.match(tpeView(data,''),/the best 2 by MSE/);
  assert.match(tpeView(data,''),/Worst in good set<\/dt><dd>2</);
  assert.match(html,/Best in good set<\/dt><dd>1</);
  assert.match(html,/model\/view<\/td><td>snapshot<\/td><td>1 of 1 share it<\/td><td>1 of 3 share it<\/td>/);
  assert.match(html,/lr<\/td><td>0.015<\/td><td>0.01 – 0.01, median 0.01<\/td><td>0.01 – 0.03, median 0.02<\/td>/);
  assert.match(html,/epochs<\/td><td>4<\/td><td>3 – 3, median 3<\/td>/);
  assert.doesNotMatch(html,/Trial 0/);
  assert.doesNotMatch(botorchView(data,''),/data-trial="e"/);
  assert.match(tpeView(data,'missing'),/Choose another proposal/);
});

test('forest view lists only forest proposals and shows what the forest split on', () => {
  const data=fixture();
  data.plan.selector='forest';
  data.trials=[{id:'a',number:1,status:'complete',mse:1.1,selector:'random',config:{}}, {id:'f',number:2,status:'running',mse:null,selector:'forest',config:{}}, {id:'g',number:3,status:'running',mse:null,selector:'botorch',config:{}}];
  const prediction={predicted_mse:1.3,uncertainty:.4,completion_probability:.6};
  const diagnostics={prediction,best_observed_mse:1.1,distribution:{predicted_mse:{edges:[1,2],counts:[3]},acquisition:{edges:[-4,-2],counts:[3]}},candidates:[{...prediction,acquisition:-2,log_improvement:-2.5,config:{lr:.1}}],importance:{score:[{name:'model/dropout',value:.4},{name:'lr',value:.1}],completion:[{name:'graph/edges/cap',value:.7}]}};
  data.optimizer.proposals=[{id:'g',number:3,selector:'botorch',diagnostics,settings:{candidates:3}},{id:'f',number:2,selector:'forest',observations:1,failed_observations:0,seconds:2,settings:{model:'random forest',candidates:3},diagnostics}];
  const html=forestView(data,'');
  assert.match(html,/Gathering history/);
  assert.match(html,/What the forest split on/);
  assert.match(html,/model\/dropout<\/span><span class="phase-track"><i style="width:100%"><\/i><\/span><strong>40.0%/);
  assert.match(html,/Completion model/);
  assert.match(html,/graph\/edges\/cap/);
  assert.match(html,/data-trial="f"/);
  assert.doesNotMatch(html,/data-trial="g"/);
  assert.match(botorchView(data,''),/Run optimizer: SMAC/);
  assert.match(botorchView(data,''),/data-trial="g"/);
  assert.doesNotMatch(botorchView(data,''),/data-trial="f"/);
});
