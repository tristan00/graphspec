**Total War ground-truth graph**

Design for loading the Total War: Warhammer III corpus into the database contract of the root README, written 13 September 2026 after examining the live corpus. Nothing here is implemented yet; section 8 lists what the loader will be and how it will be verified.

**1. Source**

The corpus lives on the PostgreSQL server at 127.0.0.1:55433 (data directory on D:), database `tw_stack`, user `tw`, no password. The `tw_stack` database on port 55432 is the legacy copy that stops on 3 September 2026 and stores state as JSON blobs; it is not used. The live schemas:

| Schema | Content |
|---|---|
| `corpus` | 69 tables. The spine is `snapshot` (268,191 rows: 222,035 decisions, 46,156 interrupts) over 5,103 campaigns, latest 7 September 2026. Every snapshot has campaign scalars (`snapshot_campaign`) and the world (`snapshot_world`, `world_army`, `world_hostile`). Decision snapshots also have their entities (`snapshot_entity` → `char_state`, `char_state_ext`, `province_state`, `campaign_state`), the offered actions (`offer`, 25.7M rows) and the action taken (`taken`). Collections that change slowly are deduplicated: a state row points at a `state_set`, whose members live in 33 `*_set_member` tables (skills, techs, unit cards, buildable, built, slot states, relations, regions, rites, missions, lord pools, items, effects, resources, stances, traits, reach, mercenaries). Side streams: `interrupt` with `interrupt_option`, `interrupt_battle_panel`, `interrupt_diplo_panel`; `diplomacy_event` (typed: deals, pair checkpoints, campaign ends); `battle`, `battle_participant`, `battle_unit` (since 7 September); `event` (incidents, dilemmas, items gained); `postmortem`. |
| `dict` | 40 identifier dictionaries (`skill` 4,328, `tech_node` 1,778, `building` 1,547, `unit` 1,254, `ancillary` 757, `region` 814, `faction` 412, `agent_subtype` 585, `effect` 186, `effect_bundle` 523, `ritual` 815, `mission` 403, `trait` 159, `edict` 111, `pooled_resource` 219, `stance` 19, `dilemma` 152, …). Each row says whether the key resolved to a reference row (`is_reference`, `ref_build_id`); `dict.family` names the reference table and column per family; `dict.enum` decodes every enumerated column. |
| `ref` | The complete game pack: 1,521 tables. The ones the graph needs: `technology_nodes` and `technology_node_links` (tech trees per faction, tiers, costs, parents), `technologies` and `technology_effects_junction`; `character_skill_nodes`, `character_skill_node_links`, `character_skill_node_sets`, `character_skill_node_set_items`, `character_skills`, `character_skill_level_to_effects_junctions` (skill trees per agent subtype with tiers, ranks, parents, per-level effects); `building_levels`, `building_chains`, `building_superchains`, `building_upgrades_junction`, `building_level_required_buildings`, `building_effects_junction`, `building_units_allowed`; `main_units` and `land_units` (cost, upkeep, caps, combat statistics); `ancillaries`, `ancillary_to_effects`, `ancillaries_required_skills`; `character_traits`, trait levels and effects; `effects` (15,064) and `effect_bundles` with `effect_bundles_to_effects_junctions`; `agents`, `agent_subtypes`, `agent_actions`; `regions`, `region_to_province_junctions`, `campaign_map_regions`; `missions`, `dilemmas`, `incidents`, `rituals`, `campaign_stances`, `provincial_initiative_records` (edicts), `pooled_resources`, `factions`, `cultures`; `loc` for names. |
| `ops` | `region_geometry` (outlines and centroids for 814 regions, no adjacency), launches, trials. |

Region adjacency is not in the pack import; it is captured in every world snapshot (`region_set_member.adjacent`). Unioned over the corpus that yields 759 regions with 3,438 directed neighbour pairs, fully symmetric, covering 530 of the 562 regions of the main map and 207 of 226 of the chaos map; that is the map graph the loader uses.

How the state keys resolve to the pack (checked on the live dictionaries): skills, tech nodes, regions, agent subtypes, rituals and pooled resources resolve 100 percent; buildings, units, ancillaries, effects, effect bundles, missions and traits resolve 100 percent by key although some `is_reference` flags are stale; factions resolve 343 of 412, the rest being engine pseudo-factions (rebels, ruins) and truncation artefacts; dilemma keys resolve after stripping the `CcoCdirEventsDilemmaChoiceDetailRecord` prefix. Row-weighted over the state tables every catalogue join is 100 percent except where the value is `none` or a rebel faction. The trees: `technology_node_sets` (33 sets; a faction maps to exactly one set through faction, subculture or culture) hold 1,843 nodes and 1,792 parent links, all inside a set; `character_skill_node_sets` (547, one per agent subtype, a few per faction affinity) hold 24,649 nodes with 20,590 links of type required or subset-required, and 513 of the 585 subtypes seen have a set, covering 99.2 percent of character rows; the collector's own observed skill prerequisites (`skill_prereq_set_member`) match those links 2,359 of 2,359. Building levels chain through `building_upgrades_junction` (3,330 links, 3,295 within a chain, 135 branching nodes). Effects: 15,064 in the pack, of which techs grant 2,329 distinct, skills 5,251, buildings 1,824, ancillaries 1,150, traits 598 and bundles 3,924; 1,185 effects are granted by two or more kinds, so the effect nodes do connect the trees (skill and tech share 415, building and skill 375, building and tech 328, ancillary and skill 391).

**2. What the graph must say**

The owner's requirement: from a snapshot the graph must fully describe the world state, the action taken there and the reward attributed to it, and the graph should be action centric. Concretely:

- **World state at a step** is every entity the collector captured at that snapshot with its scalar attributes as features, and every relationship among entities as a fact at that step: who owns which region, where every army stands, which faction is at war with whom, which skills a lord has and at which level, which buildings stand in which slot at what health, which technologies are researched or researchable, which units an army holds and their strength, which items are equipped or in the pool, which effects those techs, skills, buildings and items grant, which rites, missions and edicts are available, which regions and characters each character can reach this turn, which hostile armies are visible and how far away.
- **The structure behind the state** is the static game structure that gives those facts meaning: the technology tree (prerequisite links), the skill tree of each agent subtype (links, tiers, rank requirements), building chains (upgrade paths, prerequisites, unit unlocks), item and trait effect tables, region adjacency and province membership. It is restated at every step, as the contract requires.
- **The action** is the query of the step: its type, its acting entity, the entity it acts on, the place it happens, and its parameters. Interrupt screens (battles, occupations, dilemmas, diplomacy proposals) are actions too and get their own steps.
- **The reward** is the label of the query, computed exactly as the corpus's own trainers compute it (section 6).
- **Consequences** appear as facts at later steps: battles fought with their outcomes, diplomacy deals, items gained, regions changing owner, characters wounded or gone.

Execution and verification outcomes of the taken action (`taken.executed`, `confirmed`, `counted`, refusal, timing) never enter features, facts or row selection. Model scores stored on offers (`score`, `exploit`, `ggnn_score`) never enter either.

**3. Graphs, steps and periods**

One graph per campaign. A step is one snapshot of that campaign in `snapshot_id` order, decisions and interrupts alike; the contract's requirement of exactly one query per step is met because every snapshot carries exactly one action. The period is the game turn (`snapshot.turn`). The first bank takes the newest 100 campaigns by `first_snapshot_id`, to get the structure working; the loader takes the campaign window as a parameter, so the bank grows to the trainers' 1,000-campaign window (63,579 decisions and 10,907 interrupts, a median of 58 decisions per campaign, a maximum of 204) and then to all 5,103 once the sizes are confirmed.

Ticks: an observation of step s is visible at s and later; the query at s is the action chosen after observing s. The bank already uses observation tick 2s and choice tick 2s+1.

**4. Node types**

Entity numbers are per graph. Catalogue entities (techs, skills, buildings, …) exist in a campaign only when something in that campaign refers to them; that keeps each graph to the faction's own trees rather than the whole pack.

| Node type | Identity within a campaign | Where it comes from | Observation features (values plus presence, hashed by the infra encoder) |
|---|---|---|---|
| `faction` | `dict.faction` id | the player's faction (`snapshot_campaign`), every faction in `relation_set_member`, `war_graph_set_member`, `world_hostile`, `enemy_agent_set_member`, region owners | player: turn, income, treasury, settlements, armies, lord_level, allies, vassals, power_rank, is_researching, ll_wounded, defeated, difficulty, research_points, hero counts by type, the v31 finance block when present (net income, expenditure, upkeep, trade, tax level, food, influence). Other factions: culture and subculture from `ref.factions`, and what the world shows of them (number of visible armies, settlements owned in view). |
| `character` | (campaign, cqi) from `corpus.character` for own lords and heroes; cqi from `world_hostile`, `world_army`, `enemy_agent_set_member`, `battle_participant` for others | lords, heroes, hostile generals and agents, allied armies | own: kind (lord/hero), agent type, subtype, rank, xp, xp to next level, skill points, hp, unit count, upkeep, loyalty, authority, subterfuge, zeal, resurrection turns, action points (pct, remaining, per turn), pending recruits, x, y, acted, besieging, garrisoned, wounded, is_leader, is_agent, can_embed, is_general, has_army, in_own_territory, reach_max and the eight reach rays, raiding income. Hostiles and agents: kind (army, neutral army, hero, neutral hero), x, y, distance, units, hp, visible, armed citizenry, stance, subtype, agent type. Absent fields stay absent through presence flags. |
| `region` | `dict.region` id | own provinces (`province_state`), world regions (`region_set_member`), settlements (`settlement_set_member`, hostiles of kind settlement), ruins, reach sets, battle regions | x, y, capital, abandoned, ruin, settlement present, settlement level, public order, buildings, growth per turn, gross income, income, development points, max and free slots, has port, has walls, can set edict, garrison units when a hostile settlement row exists, complete-owner flag, climate from `ref.regions` when present |
| `province` | `dict.province` id | `province_state.province_id`, `region_set_member.province_id` | presence only; state is on regions and edges |
| `slot` | (region, slot index) | `slot_state_set_member`, `built_set_member`, `buildable_set_member`, `building_now_set_member`, `horde_slot_set_member` for horde lords (slot of a character rather than a region) | index, empty, locked, health, max health, damaged, ruined, repairing, upgrading, dismantling, queued, refund, repair cost |
| `tech` | `dict.tech_node` id | `tech_set_member`, `campaign_state.current_research_id`, and every node of the faction's `technology_node_set` (the observed tech set already equals the whole set: 74 of 74 for Cathay) | tier, indent, research points required, cost per round, food and resource cost, required parents, civil/engineering/military flags |
| `skill` | `dict.skill` id | `skill_set_member`, `hidden_skill_ids`, `hidden_skill_state_set_member`, background skills, lord-pool candidate skills, `skill_prereq_set_member`, and the whole `character_skill_node_set` of every subtype present | tier, indent, points on creation, required parents, unlocked at rank, background flag, influence cost |
| `building` | `dict.building` id (a building level) | `built_set_member`, `buildable_set_member`, `building_now_set_member`, `slot_state_set_member`, horde slots, chain neighbours through `ref.building_upgrades_junction` and `building_level_required_buildings` | level, create cost, create time, upkeep, food cost, development point cost, only in capital, can be damaged, chain and superchain as categories |
| `unit` | `dict.unit` id | `unit_card_set_member`, `recruitable_set_member`, `merc_pool_set_member`, `faction_merc_set_member`, `pending_queue_set_member`, `battle_unit`, `building_units_allowed` for present buildings | recruitment cost, upkeep, create time, campaign cap, tier, number of men, category and class, morale, armour, shield, melee attack, melee defence, charge bonus, missile strength, speed and the other `land_units` statistics, monstrous and siege flags |
| `item` | `dict.ancillary` id | `item_slot_set_member` (equipped, pool, equipped_all), `event` rows of kind ancillary gained | type, category, subcategory, legendary, transferrable, randomly dropped, uniqueness score |
| `trait` | `dict.trait` id | `trait_set_member`, `trait_progress_set_member`, lord-pool candidate traits | hidden, precedence, no-going-back level |
| `effect` | `dict.effect` id, or the `ref.effects` key when a catalogue node grants an effect the collector never observed | granted by present techs, skills, buildings, items, traits and effect bundles | category, is-positive-good, and the effect key as a category |
| `bundle` | `dict.effect_bundle` id | effect bundles on the faction, characters, forces, provinces and plagues | turns remaining is on the edge; the bundle key as category |
| `rite` | `dict.ritual` id | `rite_set_member` | category, cast time, cost, cooldown from `ref.rituals` |
| `mission` | `dict.mission` id | `mission_set_member` | category, is quest, is victory, issuer as category |
| `edict` | `dict.edict` id | `province_state.edict_ids`, selected and active edicts | key as category |
| `subtype` | `dict.agent_subtype` id | lord pools, character subtypes | agent type, is caster, cap, cost, recruitable |
| `candidate` | (campaign, subtype, candidate index) | `lord_pool_candidate` | rank, is agent, can recruit; its background skill, traits and unit through edges |
| `stance` | `dict.stance` id | `stance_set_member`, current stances | key as category |
| `resource` | `dict.pooled_resource` id | `resource_set_member` on the faction, corruption on regions | key as category |
| `dilemma` | `dict.dilemma` or `dict.incident` id | interrupts of kind dilemma, `event` rows | key as category |

**5. Relations**

Every relation is an `edge_` table with `seq`, `step`, `present` and float attributes. A fact is stated at every step where it holds; the loader emits a `present = false` row at the first step where a previously stated fact is gone, so the event history sees removals (war ended, region lost, unit destroyed, skill reset, building demolished). Static structure is stated at every step of the campaign, restricted to the catalogue nodes present in that campaign.

Ownership and position:

| Relation | From → to | Attributes | Source |
|---|---|---|---|
| `controls` | faction → character | — | own characters, `world_hostile.faction_id`, `enemy_agent_set_member` |
| `owns` | faction → region | — | `region_set_member.owner_id`, `world_army.region_owner_id` |
| `in_province` | region → province | — | static |
| `adjacent` | region → region | — | static, from observed adjacency |
| `located` | character → region | — | `char_state.region_id`, `world_army`, `world_hostile` |
| `stationed` | character → region | — | `stationed_set_member` (garrison commander) |
| `can_reach` | character → region | — | `reach_setts_true` |
| `can_reach_character` | character → character | — | `reach_chars_true` |
| `has_slot` | region → slot; character → slot for hordes | — | static per region |

Diplomacy (one relation per treaty so the relational operators can weigh them apart):

| Relation | From → to | Attributes |
|---|---|---|
| `standing` | faction → faction | standing |
| `at_war`, `allied`, `defensive_ally`, `military_ally`, `trade`, `non_aggression`, `military_access`, `vassal_of`, `excluded` | faction → faction | — |
| `war` | faction → faction | — (from `war_graph_set_member`, wars among third parties) |
| `proposed` | faction → faction | accepted, ok, success chance, gift size, one flag per treaty term, amount offered and demanded (from `diplomacy_event` and `interrupt_diplo_panel`) |

Characters:

| Relation | From → to | Attributes | Source |
|---|---|---|---|
| `skill` | character → skill | level, total levels, tier, active, locked by rank, locked by skill, locked by item | `skill_set_member` (status decoded to flags) |
| `hidden_skill` | character → skill | level, total levels | hidden skill states |
| `background` | character → skill | — | `char_state_ext.background_skill_id` |
| `skill_prerequisite` | skill → skill | — | `skill_prereq_set_member` (observed) and static links |
| `trait` | character → trait | level, points, threshold, chaos realm | `trait_set_member` |
| `trait_progress` | character → trait | points | `trait_progress_set_member` |
| `equipped` | character → item | slot index | `item_slot_set_member` of the equipped set |
| `armory` | character → item | — | `armory_item_ids` |
| `unit` | character → unit | strength pct, xp, position in the army | `unit_card_set_member`, one edge per card (parallel edges keep their own `seq`) |
| `recruitable` | character → unit | cost, disabled | `recruitable_set_member` |
| `mercenary` | character → unit | available, cost, can, pool kind | `merc_pool_set_member` |
| `recruiting` | character → unit | turns left, queue position | `pending_queue_set_member`, `pending_recruit_unit_ids` |
| `stance_option` | character → stance | active, can activate, can afford | `stance_set_member` |
| `in_stance` | character → stance | — | `char_state.stance_id` |
| `of_subtype` | character → subtype | — | `char_state.subtype_id` |
| `bundle` | character → bundle | turns remaining, force-level flag | `effect_bundle_set_member` (character and force sets) |

Provinces, slots and buildings:

| Relation | From → to | Attributes | Source |
|---|---|---|---|
| `holds` | slot → building | health, max health, damaged, ruined | `slot_state_set_member`, `built_set_member`, horde slots |
| `constructing` | slot → building | turns left, paused, queued | `building_now_set_member`, queued building of a slot |
| `can_build` | slot → building | cost, upkeep, level, is upgrade, can afford resources | `buildable_set_member` (the affordance the build action uses) |
| `upgrades_to` | building → building | — | static, `ref.building_upgrades_junction` |
| `requires_building` | building → building | — | static, `ref.building_level_required_buildings` |
| `unlocks_unit` | building → unit | — | static, `ref.building_units_allowed` |
| `edict_option` | province → edict | selected, active, can set | `province_state` edict columns |
| `corruption` | region → resource | value | `corruption_set_member` (kind 17 on provinces) |
| `bundle` | region → bundle | turns remaining, plague flag | province effect and plague bundles |
| `income` | region → resource | value per source | `income_breakdown_set_member` when present |

Faction level:

| Relation | From → to | Attributes | Source |
|---|---|---|---|
| `tech_state` | faction → tech | researched, can research, cost | `tech_set_member` |
| `researching` | faction → tech | research points | `campaign_state.current_research_id` |
| `tech_prerequisite` | tech → tech | — | static, `ref.technology_node_links` for the faction's node set; `technology_required_building_levels_junctions` adds `requires_building` tech → building |
| `rite` | faction → rite | can perform, invalid reason as code | `rite_set_member` |
| `mission` | faction → mission | status, turns remaining, is quest, is victory, completed, pending | `mission_set_member` |
| `issued_by` | mission → faction | — | mission issuer when it is a faction |
| `resource` | faction → resource | value | `resource_set_member` |
| `item_pool` | faction → item | pool index | `item_slot_set_member` of the pool set |
| `lord_pool` | faction → subtype | n candidates | `lord_pool_set_member` |
| `candidate_of` | candidate → subtype | rank, can, is agent | `lord_pool_candidate` |
| `candidate_skill`, `candidate_trait`, `candidate_unit` | candidate → skill / trait / unit | — | `lord_pool_candidate` |
| `faction_mercenary` | faction → unit | base max, current max | `faction_merc_set_member` |
| `bundle` | faction → bundle | turns remaining | `snapshot_campaign.effect_bundle_set_id` |
| `citizenry` | faction → character | — | `snapshot_world.citizenry` |

Catalogue structure (static):

| Relation | From → to | Attributes | Source |
|---|---|---|---|
| `grants` | tech / skill / building / item / trait / bundle → effect | value, scope code, skill level for skills, damaged and ruined values for buildings | `ref.technology_effects_junction`, `character_skill_level_to_effects_junctions`, `building_effects_junction`, `ancillary_to_effects`, trait effects, `effect_bundles_to_effects_junctions`; observed values from `effect_set_member` |
| `skill_link` | skill → skill | link type (required or subset required), tiers descended | `ref.character_skill_node_links` within the node sets of the subtypes present |
| `permits_skill` | subtype → skill | tier, indent | `ref.character_skill_nodes` for the subtypes present |
| `requires_skill` | item → skill | — | `ref.ancillaries_required_skills` |

Consequences (stated at the first step at or after their timestamp):

| Relation | From → to | Attributes | Source |
|---|---|---|---|
| `battle` | attacking character → defending character | attacker result, defender result, casualties, kills, hp lost, routed fraction, autoresolved, siege, naval, ambush, night, battle type | `battle`, `battle_participant` (both sides), `battle_unit` gives `fought_with` character → unit rows |
| `battle_at` | character → region | — | `battle.region_id` |
| `battle_report` | character → character | victory or defeat, casualties state | `interrupt_battle_panel` (ally and enemy cqi) for the pre-battle and result screens |
| `gained_item` | character → item | — | `event` rows of kind ancillary gained |
| `faced` | faction → dilemma | choice as code | `event` and dilemma interrupts |

**6. Queries and labels**

One query per step. `key` is the snapshot id, `period` the turn, `category` the action type.

Decision steps use `taken` joined to `dict.action`:

| Action type | Actor | Subject | Context | Query features |
|---|---|---|---|---|
| skills | character | skill | region of the character | level, total levels, tier, at max |
| stance | character | stance | region | active |
| recruit_unit, cancel_recruit, raise_dead, recruit_ror, recruit_blessed, recruit_imperial | character | unit | region | cost, pool availability, queue index, turns left |
| attack_army | character | target character | target region | target faction as category, x, y |
| attack_settlement, colonize, garrison, leave_garrison, move | character | region | region | x, y, sample index, reach rays and max for moves |
| items, item_unequip | character | item | region | pool or equipped index |
| horde_building | character | building | slot | slot index, slot empty |
| hero_action | hero | target character or region | target region | action name and ability as categories (the corpus keys carry the action name and target cqi, not the pack's `agent_actions` id), chance, agent type, target flags |
| building, building_repair, building_dismantle | region | building | slot | slot index, is upgrade, cost, upkeep, level, health, refund |
| edict | province | edict | region | can set, is selected |
| recruit_lord, recruit_hero | region | candidate | province | candidate index, n candidates, rank, agent type, type fielded |
| research | faction | tech | faction | cost, points available, in progress |
| rites | faction | rite | faction | rite index, invalid reason |
| diplomacy | faction | other faction | faction | one flag per term, gift size, standing, at war, allied, trade |
| end_turn, noop, or a decision with no taken row | faction | faction | faction | none |

Interrupt steps use `interrupt`, `interrupt_option` and the panels; the category is the interrupt kind:

| Kind | Actor | Subject | Context | Query features |
|---|---|---|---|---|
| pre_battle, battle_results | ally character | enemy character | region | chosen option as category (fight, autoresolve, retreat, dismiss), ally and enemy army counts, result state |
| occupation | faction | region | region | chosen option as category |
| dilemma, event_ack | faction | dilemma | faction | chosen option as category |
| diplomacy_proposal, diplomacy_notice, war_declared, ally_attacked, declare_war_cancel | faction | other faction | faction | chosen answer, attitude, strength ranks, demanded and offered amounts, treaty terms |

Signals per query: settlements, lord level, allies, vassals, income, treasury, armies and power rank at the step, with an observed mask of ones. They feed the signal view.

Label. The target table is `reward` on queries, with `y` exactly as `advisor/base_model.py` computes it for both rankers:

```
parts  = settlements, lord_level, allies, vassals      weights 1, 1, 1, 3
open_p(c, t) = value of part p at the first decision snapshot of turn t of campaign c   (corpus.turn_open)
y(d) = sum_p w_p * ( max over t >= turn(d) of open_p(c, t)  -  part_p at snapshot d )
```

The baseline for a decision is its own `snapshot_campaign` row; for an interrupt it is the same row of the interrupt snapshot when `state_at` is panel and the previous decision's row otherwise, following `interrupt_model.py`. Steps whose campaign has no later turn-open value for any part have no label and are still context. No step is excluded for a refused or unconfirmed execution.

Split. Held out if and only if `int(sha1("gnn3|" + campaign_key).hexdigest()[:4], 16) % 10000 < 1500`, the rule the corpus's graph ranker uses, so scores compare with the stack's own models. The split column is `validation` for held campaigns and `train` otherwise; every campaign is either wholly train or wholly validation.

**7. Sizes**

Measured on the full build of the newest 100 campaigns (13 September 2026), with every skill tree, effect grant, building chain, hero action and mission payload stated: 10,109 steps, 18.5M observations, 56.5M facts, one label per step, 16 campaigns held out (1,193 validation rows); the database is 14 GB and the bank 9.5 GB; the build took 65 minutes, nearly all of it the LadybugDB load. 200 sampled steps agree with the corpus on all 2,884 fact-count, label and split checks. The first build of this database carried NaN in the `side` attribute of every present `fought_with` fact of nine campaigns, which no code path writes and which a rebuild from the corpus did not reproduce; the bank build now refuses a present fact with a non-finite attribute, and the search runs on the rebuilt database (13 September 2026, 12:42). A fully joined single-graph variant (`graphs/join` single, every type connected, fraction 1) merges the bank into 19,550 entities over 10,109 steps in 173 s at 31 GB peak RAM in the building process.

The 1,000-campaign bank (14 September 2026): 74,486 steps, 117.7M observations, 329.8M facts with identical catalogue facts stated once, 63,464 training rows and 10,999 holdout rows; the staging is 4.5 GB and the bank 58 GB. Assembling the campaigns from the corpus took 25 minutes and the bank built from the staging in 105 s with 12 workers, 40 s of it partitioning the staged tables by graph.

Per training epoch on the separate-graph bank at history 16, 8,916 windows in 272 batches, with 8 packing processes feeding the GPU, measured on the second epoch of a two-epoch fit (13 September 2026, after the rebuild): signal 2 s, walks 4 s, sequence 6 s, tree 7 s, event graph 9 s, projection 12 s, affinity 16 s, mixed 26 s, memory 33 s and the snapshot view with relational message passing over two layers 53 s. The first epoch costs a few seconds more while each packing process loads the graphs it meets. A variant that is not yet on disk adds about 35 s for the hundred separate graphs and about 3 minutes for a fully joined one. Every view therefore completes its two to twelve epochs inside the 600-second trial budget on the separate-graph bank; the joined variants and the largest configurations are the ones the budget can still stop.

Measured per decision on the live corpus: 101 skill rows, 5.5 own entities, 5.4 armies, 6.3 hostiles, 19 regions, 10 relations, 62 techs, 31 rites, 22 lord-pool subtypes, 10 buildable rows and 6 slots per province, 5 unit cards per lord. With the static structure restated, a step carries about 1,200 observed entities and 2,500 facts; a campaign of 75 steps about 90,000 observations and 190,000 facts. The first window of 1,000 campaigns is therefore about 90M observations and 190M facts before the event-history deduplication of unchanged vectors, most of them catalogue nodes whose vector never changes. At a feature width of 128 float16 the observation matrices are about 23 GB in the bank, which is too much for a first build; the first build takes 100 campaigns (about 2 GB, 7,000 steps) and reports the exact sizes, and the window grows from there. Two reductions are available if needed and are flagged rather than taken: pruning effect nodes to those with a non-zero value on a present catalogue node, and a contract extension for facts that hold at every step so static structure is stored once rather than restated.

**8. Loader and verification**

The loader is the `totalwar/` package: `domain.py` declares the node types, relations with their attributes, the feature vocabulary and the database declaration; `reference.py` loads the pack tables and dictionaries once; `corpus.py` reads one campaign's rows; `graph.py` assembles a campaign into contract tables (static catalogue, world block per snapshot, entity block per decision carried into interrupt steps, consequences placed at the first snapshot after their time, the query and the label per step, explicit removal rows when a fact disappears); `reward.py` holds the label and split rules; `build.py` is the command; `verify.py` compares sampled steps against the corpus. It reads PostgreSQL through `graph_tuning_infra.postgres` and writes LadybugDB through the infra staging helper. Node vectors come from `graph_tuning_infra.features.encoder` over the declared field lists, and the assembler refuses any feature name outside the declaration.

The loader prunes nothing: every skill tree of every subtype the campaign touches (characters, armies, lord pools and candidates), every effect the pack grants, and the building chains closed over upgrade and requirement links are all stated; what to keep is the graph spec's decision. An attribute the corpus leaves empty on a present fact is written as zero, because attribute columns have no presence flag.

The search on this bank was launched on 13 September 2026: `python -m tuning.run run --bank artifacts/bank_totalwar --output artifacts/run_totalwar --continuous --selector tpe --random-trials 200 --seconds 600 --workers 8`, with the dashboard on that directory. The first 200 trials draw every graph, model and training parameter uniformly, including `graphs/join`; TPE then proposes from all of them.

Verification before any model runs on it:

- The contract checks of `graphspec.graphdb.check` pass, and `tuning.build` builds the bank.
- For twenty sampled steps, the facts and features at the step are compared row by row against a direct read of the corpus tables: every skill, tech, building, unit, item, relation and army row is found in the graph and nothing extra is present.
- Labels for every decision in the window equal the values `advisor/base_model.py` produces for the same decisions, and the split matches `stable_split`.
- Causality: no fact or observation at step s uses a snapshot with a larger id; battles and diplomacy events are placed at the first snapshot at or after their timestamp.
- `tuning.check_models` runs every view on the bank on the GPU; per-window packing time and memory are recorded for the largest campaign.
- The bank is built from the staging parquet, not from the LadybugDB copy: the copy of the relation tables runs at a few thousand rows a second on tables with tens of thousands of rows per node and returns 0 for some NaN attribute cells, and both are avoided by reading the staged tables partitioned by graph. `totalwar.verify` still checks the LadybugDB copy against the corpus.
- The dashboard serves every run under `artifacts/`, the page selects which one to watch, and it is running before the search starts.
