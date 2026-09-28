Victoria 3 GDP prediction: design, build state, data audit, Total War comparison, storage

25 September 2026. Written from scratch; it replaces every earlier version. It states the design in force, what is built against it, and what the recorder and its database still lack. Facts about code were read from the working trees: the recorder in D:\vic3_agent (commit f6bb5a4 plus its uncommitted tree) with the `vic3` PostgreSQL database, and this repository (commit a4bf961 plus its uncommitted tree). Database checks ran read-only under a statement timeout on one run. The data is a small development sample that changes by the hour, so nothing here is argued from counts.

Blocked: nothing.

**1. Design in force**

- One decision per pause. Every action taken in a pause, the event answers, the proposal answers and the main batch alike, belongs to that one decision. There is no time between them and they are not a sequence.
- One `block` node per pause. The step's query points at it. Each action is its own node, joined to the block and to the game entities and definitions its parameters name. Nothing is predicted per action and no action effects are summed; summing would assume the actions are independent.
- The input is the run's sequence of game states up to the pause: buildings, pops, politics, research, GDP, prices and the rest, one step per pause. Whether the model reads that sequence with a recurrence over frames, a per-entity recurrence or temporal edges is a search choice. All of them read the same sequence.
- The output is three sequences for the player country: national GDP as log GDP(t + τ) − log GDP(t), population (lower, middle and upper strata of `player_population`) as log population(t + τ) − log population(t), and the population-weighted standard of living (`country_standard_of_living_report.average_standard_of_living_by_population`) as SoL(t + τ) − SoL(t). Each has one value for each later pause of the run, at the elapsed game time τ of that pause. They are indexed by game time rather than by pause count, because pause lengths are random draws of the loop. The deficit is not part of any target.
- The targets stop five game years after the block.
- All three sequences are produced in one pass from the block's context, a learned embedding of the series and a learned encoding of τ. Nothing is autoregressive and there is no teacher forcing.
- Loss: squared error on y standardised per series, averaged over a block's points of one series, then over blocks, then over the three series, so every decision counts once whatever its number of future points and every series weighs the same.
- Score: per series, skill against persistence (y = 0), one minus the ratio of the model's weighted squared error to persistence's under the loss weights, clustered by run, since one run's blocks share a future. Trials are ranked by the mean of the three standardised validation losses; the weights are a search setting (`--objective`).

**2. Why this design, from the literature**

Every source below was read in full text.

- A block is one treatment, and the literature conditions on the whole block. In Sutton, Precup and Singh's [options](http://incompleteideas.net/papers/SPS-aij.pdf) (1999), a temporally extended action gets one outcome model over its whole duration. [MuZero](https://arxiv.org/abs/1911.08265) (2019) predicts a sequence of future rewards conditioned on the actual action sequence, with no per-action effect anywhere. The [Causal Transformer](https://arxiv.org/abs/2204.07258) (ICML 2022) takes the whole treatment sequence as one input. Its balancing loss exists only for confounded treatments, and the random advisor is not confounded (section 4). [GEARS](https://cs.stanford.edu/people/jure/pubs/gears-natbio23.pdf) (2023) predicts a set of perturbations through a knowledge graph because the additive model fails on interactions. The block node is the graph form of conditioning on the set.
- The sequence is emitted at once. On NN5, [Ben Taieb et al.](https://arxiv.org/abs/1108.3259) (2011) find multiple-output strategies consistently better than single-output ones. Their [bias and variance analysis](https://doi.org/10.1109/tnnls.2015.2411629) (2016) shows recursion accumulating error. [Chevillon](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-6419.2007.00518.x) (2007) shows the direct strategy helps most under misspecification. [Green et al.](https://arxiv.org/abs/2606.12990) (2026) show that feeding back the model's own outputs leaves the deployed predictor unidentified. MQ-RNN, TFT, PatchTST, N-HiTS and DLinear all emit the horizon in one shot, and [local projections](https://www.aeaweb.org/articles?id=10.1257%2F0002828053828518) are the same direct estimator in econometrics.
- The sequence is indexed by time. The irregular-sampling literature defines a forecast as one value per queried future time, decoded from the history representation and an encoding of the time, trained with MSE over the observed points only. Absent points cost nothing. That literature includes [GraFITi](https://arxiv.org/abs/2305.12932) (AAAI 2024), [FLD](https://arxiv.org/abs/2405.03582) (2024), [t-PatchGNN](https://proceedings.mlr.press/v235/zhang24bw.html) (ICML 2024) and [APN](https://arxiv.org/abs/2505.11250) (AAAI 2026). No basis over τ is fixed, since a basis needs a domain.
- Loss and target transform. Point forecasters in those papers train on squared error. Quantile grids are the MQ-RNN and TFT recipe for intervals, not for a point path. The log difference is the direct target of macro forecasting practice. Horizon weighting in [MuZero](https://arxiv.org/abs/1911.08265), [Benechehab et al.](https://arxiv.org/abs/2402.03146) (2024) and [ElasTST](https://arxiv.org/abs/2411.01842) (NeurIPS 2024) is never uniform per point. With an irregular τ, the rule that needs no constant is to average within a block, then over blocks.
- Evaluation per horizon against a naive forecast follows [Hyndman and Koehler](https://robjhyndman.com/papers/mase.pdf) (2006) and M4 practice.

**3. What moves GDP**

GDP is value added: the market value of what buildings produce minus what they consume, at the state's local price. The country's GDP is the sum of its states'. A building's local price blends the market price and a state price by market access.

- Capacity: building levels, the government and private construction queues with remaining work and speed, and construction speed per country. Construction efficiency per state is not recorded.
- Private capacity: the private queue is fed by the investment pool, which is fed by dividends and savings. Dividends and cash reserves are recorded per building.
- Method: the active production method per building and the goods quantities it sets. The recorder now stores every method's definition per ruleset, so a building type or a queued building reaches its goods, and an ordered method switch is defined: goods in and out, employment, other modifiers, requirements and profession ratios.
- Labour: staffing, throughput and salary per building; workforce per pop; subsistence and unemployment per state; migration as source and destination lists with weekly totals.
- Prices: market price and orders per market and goods; local price, production, consumption and net trade per state and goods. Shortage is per building and per state, not per goods.
- Trade: world market hubs, market areas and ports, shipping lanes with convoy cost, and straits with access and tolls.
- Multipliers: laws, decrees, institutions, technology, companies, charters, power blocs, devastation, turmoil and pollution. Technology is recorded for the player only.
- Demand: need weights per state, culture and goods, consumption per state and goods, and government and military goods for the player.
- Ownership: typed owners (`owner_kind` with owner country, building and company columns), so cross-country ownership resolves directly.
- Territory: state ownership, occupations, blockade, devastation, and the AI's attitudes and stances. A run ends in defeat when the player owns no state.

Some of the game's own future is known at the pause: remaining construction, colony progress, incorporation, law enactment and election dates. These are legal inputs.

**4. The random advisor**

Per pause the loop answers every pending event from the pre-action snapshot, through the first advisor. If the executor accepted an answer, it takes a mid-turn snapshot and recomputes the legal set. It then answers proposals through the first advisor and draws a batch. For each draw it picks an advisor at random, then an action name with probability proportional to its run weight times its mean combination weight, then a combination by parameter weight. Later picks that share a target, a linked object, a slot or a parameterless group with an earlier pick are excluded. Building is skipped while the government queue is backlogged past a fixed number of weeks. Then the game runs a random number of days.

Every probability is a function of the snapshot the batch was drawn from. A model that conditions on that snapshot therefore identifies effects without reweighting. What it needs is positivity: an action has to be possible under the sampler. The recorder now stores per-run action weights (`run_weight`) and parameter weights (`run_param_weight`), including overrides. Several names have weight zero and so sit outside the support: plays, war goals, primary demands, research dequeue and construction cancels. Events and proposals always go to the first advisor.

Consequences for the block design:

- Build decisions are observed only when the queue is not backlogged, so the block's context has to carry the queue's remaining work and speed. It does.
- Pairs the exclusion rules never allow in one batch are never seen together, so the model cannot learn their interaction. The block design stops that from being forced into an assumption, but it cannot observe what never occurs.
- Pauses with nothing drawn are real steps, with category `empty`. They are the untreated group under the same sampler.
- The estimand is the effect of ordering the block, not of the block taking effect. Whether actions succeeded never enters the graph.

**5. Recorded data**

- A run is one `run` row: world or vanilla map, ruleset, player country, speed, budgets, seeds, world size and starting tech and population levels, and an end reason once it ends. A run never changes country.
- Snapshots are `(run_id, seq)` with a game date and a phase: `start`, `post_action` (mid-turn or end of batch; both share the value) or `post_run`. The pre-action snapshot of a pause is the `start` or `post_run` one.
- Entities are anchored once per run by full id with generation. Attributes live in change-only history tables with a `present = false` row on removal. Status tables are written on change. Sample tables are append-only.
- Static data per ruleset covers vocabularies, extracted definitions, methods per building type and the five production method definition tables. Geometry per world covers provinces, adjacency and map files. Vanilla run geometry now exists but its correctness is unverified.
- Actions are `loop_action` rows with advisor, action code, `before_seq`, `after_seq`, status and parameters in `loop_action_param`. Event answers name the pre-action snapshot in `before_seq`. Proposal answers and the main batch name the mid-turn snapshot when one was taken, otherwise the pre-action one. Loop actions (speed, pause, unpause) are logged under the loop advisor. The wait (`run_days` with its `days`) is drawn with the batch and logged under the advisor that drew it.
- Pops are transient: the game splits and merges them with new ids. The persistent unit is the stratum, meaning the pops of one state, type, culture and religion.

**6. Leakage rules**

- `status`, `detail`, `ms`, `logged_at` and `after_seq` are decided after the choice and enter nothing: features, edges, sample selection, structure or splits.
- The step is the pre-action snapshot. Mid-turn and end-of-batch snapshots are never read. The mid-turn snapshot follows an executor outcome, so reading it would leak.
- Loop-advisor rows are dropped. The wait drawn with a batch is an action of its block, in group `time`, with its days as a feature; it does not set the block's category, so a pause where only the wait was drawn stays `empty`.
- Run facts that are future or wall-clock are excluded: end reason and detail, timestamps, heartbeat and progress columns, seeds, world identity, budgets and the snapshot's wall-clock columns. World size, speed and starting levels are legal.
- The split is by whole run, from a salted hash of the run id.
- Truncation: runs end on the game-year budget, the wall-clock budget, defeat, a stuck interface, a crash or a stall. Points past the end are absent and cost nothing. The mean is unbiased only if censoring does not depend on the future path given the inputs. Runs ended by game years are the clean subset for checking that. Defeat leaves y undefined, and those points are absent too. A hazard head for defeat is not built.

**7. The adapter as built**

The `vic3` package reads finished runs and writes contract tables through `Stage`. It opens one read-only connection per worker, runs every statement under a timeout and writes nothing to the database.

- Step: one per pause, at the pre-action snapshot. Each non-loop action goes to the last pre-action snapshot at or before its `before_seq`. The closing snapshot is a step only if it has actions, but it still supplies label points. The period is the game date.
- Block: one node per step, `block:<seq>`, observed at that step only. Features: turn, action count, event-answer count and days since run start.
- Query: actor and context are the player country and the subject is the block. The query vector is empty. The query signal is log GDP, log population and standard of living at the pre-action snapshot with their observed masks; only the `signal` model reads it, as the baseline that sees nothing but the three series. The graph models read the same values from the country's report features. Category: the group of the first action that is not an event answer, `events` when there are only event answers, `empty` when there are none.
- Node types: country, state, market, building, queue entry, company, law, stratum, world market hub, strategic region; block and action; interest group, character, political movement, power bloc and war; and the definition nodes goods, building type, production method, law type, technology, decree type and institution type.
- Relations: `in_block` and `targets` for actions, with no `orders`.
  - Economy: `owns`, `adjacent_country`, `import_tariff`, `export_tariff`, `market_owner`, `market_goods`, `in_market`, `local_price`, `produces_state`, `consumes_state`, `produces`, `consumes`, `uses`, `in_state`, `queued`, `will_be`, `runs`, `owned_by`, `needs`.
  - Policy: `has_law`, `enacting`, `researched`, `researching`, `decreed`, `institution`, `unlocks`, `pending`, `of_type`.
  - Space and population: `lives_in`, `works_in`, `migrates_to`, `adjacent`, `hub`, `lane`, `strait`.
  - Politics: `represents`, `leads`, `rules`, `supports`, `movement_of`, `in_bloc`, `in_war`, `occupies`, one relation per attitude value and one per stance.
  - Every fact is restated at every step, and removals are derived.
- Targets: an action's integer parameters reach country, state, building and company. Law and institution parameters resolve to their type, and text parameters reach definition nodes.
- Labels: table `future` (`id`, `y`, `split`, `tau`, `series`) with edge table `future_of` to the query. `series` is `gdp`, `population` or `sol`. `tau` is elapsed game days and must be positive. It is capped at 1825 days. Points of a series come from every later pre-action snapshot where that series is measured, and at a repeated date only the first counts. The library reads the table as level `path`.
- Split: a blake2b hash of the run id, about a quarter of runs to validation.
- `vic3.verify` checks the contract's structural rules; that the subject is the step's block; that actor and context are one country; that every observed action is `in_block` with its own block; that labels are finite and tau positive; and, against the database, the owned-state count and the GDP signal.

**8. The library as built**

- Supervision (`tuning/build.py`) takes many points per query with finite positive τ and a series name, one point per query, series and τ, and one split per query. Standardisation uses the mean and deviation of each series' train points. The registry records the series names.
- The bank carries τ, the series and each point's query. Batching carries them into the model.
- Head (`tuning/layers.py`): the query's representation, joined with a learned time encoding of τ and a learned series embedding, through a two-layer MLP, one value per point. The encoding is a search choice among log-cosine, cosine and log-linear. In the snapshot model, the query's representation is the sequence frame together with actor, subject (the block) and context.
- Loss (`tuning/training.py`): squared error on per-series standardised y, averaged per query and series, then over queries, then over series.
- Score: each series gets `val_loss` (query-weighted MSE over its training variance), raw MSE and persistence skill `val_r2`. `tuning/objective.py` blends the series losses with the run's `--objective` weights into the ranked score, used by every selector, the report and the dashboard. Predictions store the row, prediction, target, τ and series.
- The suite passes. With four workers it takes about fourteen seconds.

Built data: the `vic3-block-4y` dataset and bank from runs 1 and 4, with the four-year cap, verified with no problems.

**9. Open work, in order**

1. Suite time under ten seconds. It stands at about fourteen with four workers. The remaining pass defers heavy imports so each worker pays only for its own tests.
2. Fallbacks and simplifications in the adapter:
   - Ideology and commander-rank names fall back to the raw code.
   - A missing region shape falls back to an empty dict.
   - `present` defaults to true when absent.
   - The database comparison in verify is optional.
   - `actions_at` recomputes state it already has.
   - The query vector is always empty and could be dropped.
3. Verify does not check the four-year cap or date uniqueness. The bank builder catches duplicates later.
4. Five unmodelled parameter kinds name node types that already exist: character, interest group, political movement, war and diplomatic play. Their actions could get `targets` edges. Formations, journal entries, treaties and amendments have no node types.
5. Edge attributes reach only the `relational` spatial mode. Every other mode ignores them, and the snapshot model's final layer is `mean`.
6. The declarative builder of PLAN.md is not built. It would move restating and removal derivation into the library. Restating is the reason per-goods relations such as `local_price` dominate the fact count.
7. Stale outputs: the uncapped `vic3-block` bank and the `vic3-block-1`, `vic3-run2` and `vic3-run4` datasets predate the current design.

Later, not in the first build: a defeat hazard head, a known-future channel from the game's scheduled facts, auxiliary per-node next-step targets, and the symbolic-order variant, which the production method tables now make possible.

**10. Recorder data issues**

Resolved:

- Production method definitions: `production_method_goods`, `_employment`, `_modifier`, `_requirement` and `_profession_ratio`, per ruleset. Requirement keys are text with no foreign key to technology, and modifiers are uncoded text.
- Typed ownership: `owner_kind` with owner country, building and company columns and foreign keys. No constraint ties the kind to which owner column is set. `owner_company` is null for regional headquarters.
- Snapshot phase, entity generations, change-only history, append-only samples, run end reasons with heartbeat, vanilla geometry in code, and shipping lane endpoints.

Partly resolved:

- Sampler record. Weights per run and per parameter are stored. The legal set per pause, the touched set before each draw and the option code version are not, and there is no per-decision row.
- Vanilla. A vanilla run exists with its geometry, but that geometry is unverified.
- Run facts. World size and starting levels are now on the run row. Crashed runs no longer count toward a round.

Open:

1. Legal set per pause, touched set and code version: the one thing Total War records and this recorder does not.
2. Series samples are dated only at the buffer head, and the sample rate is dropped. The metric reports already carry GDP at every pre-action snapshot, so the series may not be needed at all.
3. Dropped and wanted: construction efficiency per state, the coastal flag, trade centre identity and per-goods shortage.
4. Action parameters that name entities are raw integers with no foreign key.
5. Ordinal parameters (event options, army options, templates, journal buttons) are not resolved to targets at logging.
6. Visibility mismatches:
   - `invasion` is declared public but captured through the player's wars.
   - Strait endpoints are player-only under a public table.
   - Company charters are player-only columns in a public table.
7. State adjacency is not stored. It is derivable from province adjacency, and the adapter derives it.
8. Pops are stored per pop. The adapter aggregates them to strata, and aggregating at capture would also remove the largest storage cost.
9. The annualisation between building flows and state GDP is unverified.
10. `catalogue_action` is global and rewritten per process, although `run_weight` now covers the per-run action names.
11. Other countries' technology is not recorded.
12. `state_goods_supply` is a player-relative flag whose name reads like production.
13. Censoring by wall clock, crash or stall can depend on the economy through game speed. Game-year endings are the clean subset.
14. There is no save-and-reload branching for paired counterfactuals, which remains the largest available variance reduction.
15. Fallbacks in the recording path:
    - Unknown enum codes are inserted silently.
    - `law_phase` has codes without names.
    - Open vocabularies insert unknown names.
    - `PARKED` is always empty.
    - `DROPPED_FIELDS` is imported by nothing.
    - Pause and speed columns are nullable.
16. The mid-turn snapshot is taken only when an event answer was accepted. The adapter no longer reads it, so this now matters only to anyone else who uses that snapshot.

**11. Comparison with Total War**

| Area | Victoria 3 | Total War | Better |
|---|---|---|---|
| Decision unit | one block per pause, holding every action taken in it | one decision per action opportunity, interrupts as extra steps | different by design; both give the model one query per decision |
| State per step | full world state at the pre-action snapshot | the player's entity block | Vic3 |
| Outcome recorded | executor report only, never read; no verification | execution and confirmation flags on the taken row | Vic3: nothing about success is near the state tables |
| Legal actions | computed and discarded | offer rows keep the legal set with scores | TW |
| Propensities | run and parameter weights stored; legal set not | arm weights and eligible set stored | TW until the per-decision row exists; Vic3 then, since it is randomised |
| Identity | full id with generation, anchored per run | integer surrogate dictionaries | equal |
| Catalogue | vocabularies and definitions per ruleset, now with production method effects | the whole game pack with a build id and collector version | TW on versioning, Vic3 on per-run linkage |
| Changing state | change-only rows with removal rows | content-hashed shared sets | equal in bytes; Vic3 simpler to read |
| Geography | provinces, adjacency and centroids per world | region centroids and observed adjacency | Vic3 where a world exists |
| Time | game date, irregular pause lengths | regular turn index | Vic3 for a time-indexed path target; TW simpler |
| Target | future GDP path per block, time-indexed, capped | maximum over the remaining campaign, which leaks the truncation time | Vic3 |
| Other countries | public tables and metric reports for every country | the player's view | Vic3 |
| Run bookkeeping | run row with world, budgets, end reason, heartbeat | campaign row, collector version | Vic3 |

Vic3 could take from Total War the stored legal set per decision, and a collector version per snapshot beyond the schema digest. Total War could take the stored phase, full state per step, the run row with end reason, and randomised play for identification. Neither recorder makes the other's graph easier: the contract is the shared layer, and the declarative builder is the shared fix.

**12. Database efficiency**

Both databases share one PostgreSQL 17 server with default autovacuum and no partitioning.

Victoria 3:

- Every one of the recorder's tables has a btree primary key next to a BRIN on run and snapshot. On the wide history and per-goods tables the key approaches the size of the heap. It guards nothing the writer does not already guard in memory, and foreign keys point only at anchor tables.
- The largest group is the pop tables, because pops are transient.
- Every measure is float8, including shares and rates.
- Each generated world stores its map images.
- `state_pop_need` has the widest key. It may be derivable from culture and prices.
- The data tables are append-only, so update bloat is not a concern.

Ranked by the cost each removes:

1. Drop the primary keys on history, child, status and sample tables and keep them on anchors, vocabularies and meta tables.
2. Aggregate pops to strata at capture.
3. Decide `state_pop_need` by the derivability question.
4. Use float4 for shares, rates and weights.
5. Stop storing map images, since the geometry is already extracted.
6. Partition by run range once purging becomes routine.

Total War: a read-only corpus dominated by the offer table, whose action index is larger than its primary key. Many reference and corpus indexes are never scanned, and the analytics tables are rarely read. One-off drops of those indexes, the derivable rank columns and the analytics tables recover most of the space.

**13. Open questions for the owner**

- Whether the per-decision row with the legal set is logged from now on only, or the option code is versioned so past runs can be reconstructed.
- Whether pops are aggregated at capture.
- Whether the series tables are kept, given that the metric reports carry every label.
- Whether save-and-reload branching for paired counterfactuals is worth building.
- Whether the history primary keys are dropped.
