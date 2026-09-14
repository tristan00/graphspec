**Model audit and plan**

Written 12 September 2026 from a line-by-line reading of `tuning/*_models.py`, `tuning/layers.py`, `tuning/message_net.py`, `tuning/turns.py`, `tuning/turn_inputs.py`, `tuning/roster.py`, `tuning/selector.py` and the packing code, compared with the cited papers. Three independent audits (snapshot family, memory and event family, neighbour-sequence and walk family) fed this document.

**1. What the thirty models actually are**

Verdicts: faithful means the defining computation of the paper is present; loose means the shape is there but a defining detail is replaced; name only means the name explains little of what runs.

| Id | Claimed | Verdict | What runs |
|---|---|---|---|
| M01 | Relational snapshot GRU | no paper | relational message passing over the window's observation graph, a query feedback round per frame, one graph summary vector per step through `nn.GRU` |
| M02 | Relational snapshot LSTM | no paper | M01 with `nn.LSTM` |
| M03 | Relational snapshot Transformer | no paper | M01 with causal self-attention over frames |
| M04 | GCN-GRU | loose | symmetric-normalised aggregation plus an extra root weight and LayerNorm, then a per-entity `GRUCell` |
| M05 | GraphSAGE-GRU | faithful | mean aggregator in concat form, then a per-entity `GRUCell` |
| M06 | RGCN-GRU | loose | per-relation weights but one mean over all relations, attributes as messages, then a per-entity `GRUCell` |
| M07 | RGCN-TCN | faithful to gated TCN | M06 with gated dilated causal convolutions |
| M08 | T-GCN | loose | graph convolution inside every GRU gate (the GConvGRU cell); the paper convolves the input only |
| M09 | DCRNN | faithful cell | M08 with K-hop forward and reverse random-walk diffusion; no encoder-decoder |
| M10 | EvolveGCN-H | faithful | top-k summary drives a matrix GRU over the GCN weights, per sequence |
| M11 | DySAT | loose | single-head dot-product structural attention, sinusoidal positions, added feed-forward block |
| M12 | MPNN over derived nodes | faithful to the template | edge-conditioned messages with GRU vertex update on the mixed graph |
| M13 | Entity-coupling gated | no paper | gated message passing over an entity's dated copies through coupling and boundary edges |
| M14 | Entity-coupling attention | no paper | M13 with additive attention |
| M15 | Event-graph gated | no paper | gated message passing over member events linked by the event DAG, with per-entity anchors |
| M16 | Event-graph attention | no paper | M15 with attention |
| M17 | TGAT | faithful, single head | recursive temporal-neighbour attention with time-of-interaction cutoffs; `heads` not read |
| M18 | Typed TGN | loose | GRU mailbox memory with mean aggregation, one attention layer without edge features |
| M19 | JODIE | loose | mutual RNN states and a log-time projection; relation events ignored; no trajectory loss |
| M20 | DyRep | loose | two intensity-weighted processes, but attention over the receiver's current snapshot rather than the partner's neighbourhood; no survival loss |
| M21 | GraphMixer | loose | MLP-Mixer over sampled tokens with learnable log-time encoding; no link features; oldest-first order |
| M22 | DyGFormer | loose | co-occurrence counts, patching, joint Transformer; root-token readout |
| M23 | HyperEvent | faithful | the twelve pair-relative structural features and a Transformer |
| M24 | TPNet | faithful core | seeded random-projection walk matrices, decay per hop order, Gram features |
| M25 | CAWN | faithful | causal biased walks, two-endpoint hitting-count identities, BiLSTM, attention pooling |
| M26 | TNCN | faithful derivative | TGN memory plus Hadamard pair term and one-hop common neighbours |
| M27 | DyGMamba | faithful, simplified | node-level and pair-time selective SSMs with the cross-node selection; no patching |
| M28 | NAViS | faithful update, different task | the four gating equations over category affinities; regression instead of ranking; packed `global_affinity` never read |
| M29 | CTAN | loose | antisymmetric Euler integration, but the neighbourhood term is a mean over snapshot edges and no elapsed-time input |
| M30 | MTGNN | faithful graph learning and mix-hop | single gated dilated convolution instead of the inception module; assumes one prediction per frame |

Systematic deviations shared by whole families: every time encoding is `cos(log1p(gap) w + b)` with learnable `w`, where the papers use linear time and GraphMixer insists on a fixed encoding; the neighbour-sequence and walk paths never tokenise relation codes or edge attributes; every readout is a query-conditioned regression MLP where the papers do link classification or ranking; several models receive parameters they never read (`dropout` for M15 to M30 except M22 and M23, `heads` for M17, `diffusion_hops` for M08, `neighbors` for every snapshot model).

**2. The parts they are made of**

The thirty names hide about twenty building blocks, most of them shared. Grouped by the role they play:

Input views, one per packed block: the snapshot graph of observation nodes and relation edges per step; the coupling graph of an entity's dated copies; the derived-node graph (mixed); the event stream with per-entity anchors, optionally with DAG links; per-endpoint neighbour token sequences, optionally with the recursive tree; causal walks; category affinities; signal channels.

Encoders: `Linear(x)` plus kind embedding, with or without LayerNorm and SiLU; `TimeEncoding`; relation embedding; structural features (two co-occurrence counts, twelve pair-relative features, walk hitting counts); edge attributes as `sign x log1p`.

Spatial propagation within a snapshot: none; mean; symmetric-normalised; per-relation weights with attributes; dot-product attention; Chebyshev polynomial filters of the frame Laplacian (ChebNet); relation-aware dot-product attention with a relation embedding in keys and messages. All wrapped in `root(h) + update`, SiLU, dropout, residual, LayerNorm.

Temporal operators: per-entity `GRUCell`, `LSTM`, gated dilated TCN, causal self-attention; the graph cell with plain or K-hop diffusion convolution inside the gates; evolving GCN weights; frame-level sequence over graph summary vectors; message passing over temporal edges (gated or attention); event-driven memory (GRU mailbox, mutual RNN, intensity-weighted tanh, antisymmetric Euler); random-projection walk matrices; sequence models over neighbour tokens (MLP-Mixer, Transformer with patching, selective SSM with a pair-time branch, recursive attention); BiLSTM over walks with attention pooling; gated affinity state; learned top-k channel adjacency with mix-hop and gated convolution.

Readouts: states of actor, subject and context entities; pooled reachable events; pair terms (Hadamard product, common-neighbour sums, occurrence counts); the projected query; period context by gate (`EndContext`), by extra tokens, or by extra edges; a two-layer MLP to one value.

Pairs that differ by exactly one switch: M01/M02/M03, M04/M05/M06, M06/M07, M08/M09, M13/M14, M15/M16, M22/M23. One class serves five models through `family` branches in `memory_models.py` and another serves four in `history_models.py`.

**3. Graph considerations per view**

| View | Representation group and spec parameters | Graph assumptions |
|---|---|---|
| snapshot | none beyond the base arrays; `relations/direction`, node filters and derived relations change its edges | node kinds; relation attributes only read in relational mode; the M01 to M03 decision round needs queries, otherwise it degenerates to pooled means |
| coupling | `coupling/*` (scheme, unit, lags, window, cap, periods) | none |
| mixed | `mixed/*` (changes, trace, bundles) | queries add action nodes |
| events, DAG | `events/*` (kinds, link, lags, window, cap, periods) | relation removals appear as status; observation events carry no partner |
| neighbour sequences, tree | `neighbors/*` (kinds, removals, order, decay) | sampled at the query cutoff, not at each event's time (matters for HyperEvent's two-hop features) |
| walks | `neighbors/kinds`, `neighbors/removals`, `walks/*` (bias, window, fanout); 32 walks of 4 hops stored | none |
| affinities | none | queries with categories; query or graph target |
| signals | none | `signal` channels present; one prediction per frame |

The packing layer already derives which blocks a model needs from `roster.INPUTS`; the graph spec space already restricts itself to the groups a model reads. Those two mechanisms are the seam the plan builds on.

**4. What is wrong with the current setup**

- Names carry the authority of papers that the code follows only loosely, and the README's one-line descriptions sometimes describe the code and sometimes the paper. The reader cannot tell from `M18 Typed TGN` that the aggregator is a mean, that keys have no edge features, or that there is one attention layer.
- The search treats thirty names as categorical alternatives. TPE cannot learn that M04 and M05 share everything but the aggregation, or that M22 and M23 differ by one feature block, so it wastes trials rediscovering the structure the code already has.
- Many switches that matter are not tunable (time encoding form, aggregator, attention type, readout composition, period-context mode), while some tunable parameters are dead for the models that receive them.
- The families that are honest house designs (M01 to M03, M13 to M16) are presented in the same list as paper reproductions.

**5. Plan: a parameterised model space**

Goal: the tuner proposes model parameters that order the construction of the model, the way the graph spec orders the construction of the variant. No paper names in the space; components are named by what they compute, and the papers become citations against components with their deviations listed.

5.1 Reuse the spec machinery. `graphspec/spec.py` already has conditional parameters, `draw`, `canonical`, `digest` and `summary`. The model space becomes a second `space()` over `model/...` parameters built with the same `Parameter` type, so TPE, the selector, the trial directory names and the dashboard treat model and graph parameters identically.

5.2 Views as the top of the tree. `model/view` selects the input view: `snapshot`, `coupling`, `mixed`, `events`, `sequence`, `walks`, `affinity`, `signal`. Each view fixes the packed blocks it needs (today's `roster.INPUTS`) and the representation groups the graph space must offer (today's `roster.REPRESENTATIONS`). Views that need queries, signals or categories are offered only when the bank has them, as `roster.models` does now.

5.3 Conditional parameters per view, taken directly from the switches found in the audit:

- snapshot: `spatial/mode` (none, mean, normalised, relational, attention), `spatial/layers`, `spatial/attributes` (on/off); `temporal/operator` (entity_gru, entity_lstm, entity_tcn, entity_attention, graph_cell, evolving, frame_sequence); `temporal/diffusion_hops` for graph_cell; `frame_sequence/operator` (gru, lstm, attention) and `frame_sequence/feedback` (on/off) for the frame path; `readout/entities` (actor, actor+subject, actor+subject+context).
- coupling and events (temporal-edge message passing): `message/mode` (gated, attention), `message/layers`, `anchors` (events only), `readout` (pooled events or entity states).
- events with memory: `memory/update` (gru_mailbox, mutual_rnn, intensity, antisymmetric_euler), `memory/aggregator` (mean, last), `memory/message` flags (partner_state, time, relation, status), `embedding/layers` (0, 1, 2) and `embedding/heads`, `pair/terms` (none, hadamard, common_neighbours, both), `projection/hops`, `projection/width`, `projection/decay` for the walk-matrix branch, `euler/steps`, `euler/damping`.
- sequence: `token/features` flags (observation, time, relation, co_occurrence, structure), `sequence/operator` (mixer, transformer, ssm, recursive_attention), `transformer/heads`, `transformer/patch`, `ssm/state`, `pair_time` (on/off), `tree/depth`, `readout` (root_tokens, mean).
- walks: `walks/count`, `walks/hops`, `walk/encoder` (bilstm, transformer), `walk/pooling` (attention, mean), `identity` (on/off).
- mixed: `aggregation` (mean, attention), `update` (gru, residual).
- affinity: `buffer`, `global` (batch_buffer, packed_global_affinity).
- signal: `topk`, `mix_hops`, `temporal/inception` (on/off).
- shared: `hidden`, `dropout`, `time/encoding` (log_learnable, log_fixed, linear_learnable), `context/mode` (gate, tokens, edges, none), `head/layers`.

Every parameter is read by the component it conditions; the dead parameters of today disappear by construction.

5.4 Honest names. `describe(spec)` renders a name from the active parameters, for example `snapshot relational×2 → graph_cell diffusion 2 → actor+subject+context` or `sequence observation+time+co_occurrence → transformer 4 heads patch 2 → root tokens`. Trial directories, `results.csv`, `progress.md` and the dashboard use that string. The README lists components with their equations and cites papers per component with the deviations stated (for example: recursive attention, single head, additive time and relation terms, log time; after TGAT).

5.5 Component library. Move the building blocks out of the three model files into `tuning/components/` grouped by role (encoders, spatial, temporal, memory, sequence, readout), each a small module with one clear computation and no `family` branches. A `build(model_spec, registry, categories)` assembles them. `MemoryModel` and `HistoryModel` dissolve into components plus parameters.

5.6 Packing follows the spec. The blocks a model needs are computed from its view and features rather than from `roster.INPUTS`, so adding a component cannot silently miss its input. Structural features that cost quadratic time (`structure`, `co_occurrence`) are only computed when the spec turns them on, as now.

5.7 Search. The selector draws `model/...` parameters through the same conditional `draw` as `graph/...`; the graph space offers only the representation groups the chosen view reads. Random and TPE selectors need no other change. The plan file, selection records and reports carry the model spec and its digest exactly as they carry the graph spec.

5.8 Verification. Tests per component (forward, backward, shape) and per view (every switch builds and runs on the synthetic banks at all three target levels; batch of windows equals the concatenation of single windows; a truncated window changes nothing before the cut). Anchor configurations: for each of the thirty current models, the setting in the new space that reproduces its computation, checked by parameter count and by identical outputs under copied weights where the layout allows; those anchors document what each old name meant and let a search resume from an old result. The suite stays under 20 seconds by using the tiny synthetic bank.

5.9 Order of work and gates. (1) Write the model space and `describe` with the anchors as tests of the space alone, no models yet. (2) Extract components without changing behaviour, verified by identical outputs for all thirty models under copied weights. (3) Replace the three model files with the builder; verify anchors. (4) Switch packing to spec-derived blocks; verify with the packing equality harness. (5) Switch the selector, reports and dashboard to the model spec; run a short search on the synthetic banks. (6) Rewrite `tuning/README.md` Models into component tables with cited deviations, and delete the names. Each step is a commit; nothing is deleted before its replacement is verified.

5.10 Decisions this plan leaves to you. Whether to keep the house designs (frame-level sequence with query feedback, coupling and event-DAG message passing) as views: they are honest and cheap, and their audits found no defects. Whether the regression head stays the only task, which removes the paper losses (ranking, link classification, survival) from scope. Whether the fixed-time-encoding variants and linear time are worth their extra parameters, or whether the log-learnable form should stay the only one.

**6. Status after the rebuild**

The thirty named models are gone. `tuning/model_spec.py` holds the parameter space (`view` at the root, one conditional switch per component, described in `tuning/README.md`), `tuning/models.py` builds a network from a spec, and the components live in `snapshot.py`, `message_net.py`, `events.py`, `neighbors.py`, `affinity.py` and `signal.py` without family branches. Packing derives the blocks it fills from the spec, the graph space derives its representation groups from the blocks, the selector draws `model/...`, training and `graph/...` parameters through the same conditional machinery, and every trial, result row and dashboard row is named by `describe(spec)`. No new component was added; every switch selects code that the old models already ran.

Every old model is one point of the space. Verified on the synthetic dev bank with the old implementation held in a scratch copy: parameter counts equal, and with the old weights copied into the new network the outputs on a four-graph batch are equal to float tolerance, for all thirty configurations, with and without period context (60 of 60 cases). The coverage check `tuning.check_models` ran 69 configurations (one per structural switch setting, including the switches that only appear under another switch) on the dev bank with and without period context: all forward, backward and reload, and no parameter is left without a gradient. Two adversarial reviews of the diff each ran their own checks: the presets also agree in training mode with dropout active and on a node-level bank with many rows per window; the review found and the code now fixes a pair-term configuration that packed no neighbours, a dashboard crash on a baseline without a model spec, and documentation that overstated where time encodings and the projected query are used.

| Old id | Spec |
|---|---|
| M01, M02, M03 | `snapshot` spatial relational, temporal graph_sequence, sequence gru / lstm / attention |
| M04, M05, M06 | `snapshot` spatial normalised / mean / relational, temporal entity_gru |
| M07 | `snapshot` spatial relational, temporal entity_tcn, temporal_layers = old layers + 1 |
| M08, M09 | `snapshot` spatial none, temporal graph_cell, diffusion_hops 0 / K |
| M10 | `snapshot` spatial none, temporal evolving |
| M11 | `snapshot` spatial attention, temporal entity_attention |
| M12 | `mixed` aggregation mean, update gru |
| M13, M14 | `snapshot` spatial relational, temporal coupling_edges, message gated / attention |
| M15, M16 | `event_graph` message gated / attention |
| M17 | `tree` |
| M18, M26 | `memory` update mailbox_gru, embedding attention, pair_terms off / on |
| M19 | `memory` update mutual_rnn |
| M20 | `memory` update intensity, embedding attention |
| M21 | `sequence` operator mixer, structure none |
| M22, M23 | `sequence` operator transformer, structure occurrence with patch 2 / overlap with patch 1 |
| M24 | `projection` |
| M25 | `walks` |
| M27 | `sequence` operator ssm, structure none |
| M28 | `affinity` |
| M29 | `memory` update antisymmetric, embedding state |
| M30 | `signal` |

Two behaviours changed deliberately. `sequence/patch` 1 now means no patch projection, where the old M22 with patch 1 still applied a square linear map. The `mixed` and `event_graph` views no longer pack the snapshot block they never read. Combinations the old roster could not express are now reachable, for example a relational spatial stack before the graph cell, attention embedding on the antisymmetric memory, structural features on the mixer, or attention aggregation with a residual update on the derived-node graph; the coverage check exercises one configuration per switch setting, so each of those runs, but none has been trained.

The random selector draws views uniformly, so the snapshot view's thirty-five spatial and temporal combinations share one tenth of the random trials where the old roster gave them fourteen thirtieths; TPE conditions on the drawn view and is unaffected.
