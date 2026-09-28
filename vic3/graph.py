import logging
import math
import time
from bisect import bisect_right

from .domain import (
    CATEGORIES,
    CHOICE_PARAMS,
    ENACTING_PHASES,
    ENTITY_TARGETS,
    EVENT,
    GROUP,
    HISTORY,
    HORIZON_DAYS,
    HOURS_PER_DAY,
    LAW_PRIVATE,
    LOOP_ADVISOR,
    NODES,
    NUMBERS,
    OWNER_KINDS,
    PLAYER,
    POLICY,
    PRE_ACTION,
    RELATIONS,
    REPORTS,
    RESOLVED_TARGETS,
    SERIES,
    STRATA,
    STRATUM_MEANS,
    STRATUM_SUMS,
    SUBJECTS,
    TEXT_TARGETS,
    UNMODELLED_PARAMS,
    field,
)
from .tables import chunk_tables, label_tables, node_tables, removal_values

LOG = logging.getLogger(__name__)
CHUNK = 8


def finite(value):
    return value if value is not None and math.isfinite(value) else None


def logarithm(value):
    return math.log(value) if value is not None and value > 0 and math.isfinite(value) else None


class Run:
    def __init__(self, encode, features, run, data, names, columns, parameters, allowed):
        self.encode, self.features, self.run = encode, features, run
        self.snapshots = {row["seq"]: row for row in data["snapshots"]}
        self.tables = data["tables"]
        self.names, self.columns, self.parameters, self.allowed = names, columns, parameters, allowed
        self.moments = [row["seq"] for row in data["snapshots"] if row["phase"] in PRE_ACTION]
        if not self.moments:
            raise ValueError("run %d has no pre-action snapshots" % run["run_id"])
        self.start = self.snapshots[self.moments[0]]["game_date"]
        self.actions = {seq: [] for seq in self.moments}
        for row in data["actions"]:
            if row["advisor"] == LOOP_ADVISOR:
                continue
            pause = bisect_right(self.moments, row["before_seq"]) - 1
            if pause < 0:
                raise ValueError("action %d precedes the first pre-action snapshot" % row["n"])
            self.actions[self.moments[pause]].append(row)
        self.steps = [seq for seq in self.moments if seq != self.moments[-1] or self.actions[seq]]
        if not self.steps:
            raise ValueError("run %d has no decision moments" % run["run_id"])
        self.entities, self.kinds, self.keys, self.static = {}, [], [], {}
        self.player = self.entity("country", run["player_country"])
        self.unlocks = data["unlocks"]
        self.shapes, self.borders = geometry(data["world"], names["terrain"])

    def entity(self, kind, ident):
        if kind not in NODES or ident is None:
            raise ValueError("node requires a declared kind and a non-null key: %s %r" % (kind, ident))
        key = kind + ":" + str(ident)
        found = self.entities.get(key)
        if found is None:
            found = len(self.kinds)
            self.entities[key] = found
            self.kinds.append(kind)
            self.keys.append(key)
        return found

    def code(self, kind, code):
        name = self.names[kind].get(code)
        if name is None:
            raise ValueError("unknown %s code %r" % (kind, code))
        return self.named(kind, name)

    def named(self, kind, name):
        node = self.entity(kind, name)
        self.static[node] = dict(key=name)
        return node

    def vector(self, kind, data):
        unknown = set(data) - self.allowed
        if unknown:
            raise ValueError("undeclared features for %s: %s" % (kind, ", ".join(sorted(unknown))))
        return self.encode(kind, data)

    def current(self, seq):
        return {table: versions.advance(seq) for table, versions in self.tables.items()}

    def build(self, stage, counter, index, name, held_out):
        started = time.perf_counter()
        previous, dates, measured = {}, [], []
        observations, fact_rows, queries = [], {}, []
        for step, seq in enumerate(self.steps):
            state = self.current(seq)
            features, facts = {}, {}
            self.world(state, features, facts)
            self.layers(state, features, facts)
            block = self.actions_at(step, seq, features, facts)
            for relation, items in facts.items():
                for source, target in items:
                    for node in (source, target):
                        if node not in features:
                            if node not in self.static:
                                raise ValueError(
                                    "%s fact names an unobserved node %s" % (relation, self.keys[node])
                                )
                            features[node] = self.static[node]
            for position, node in enumerate(sorted(features)):
                observations.append((node, step, position, self.vector(self.kinds[node], features[node])))
            present = {relation: set(items) for relation, items in facts.items()}
            counted = 0
            for relation in RELATIONS:
                for (source, target), values in facts.get(relation, {}).items():
                    fact_rows.setdefault(relation, []).append((source, target, counted, step, True) + values)
                    counted += 1
            counted = 0
            for relation, keys in previous.items():
                for source, target in sorted(keys - present.get(relation, set())):
                    fact_rows.setdefault(relation, []).append(
                        (source, target, counted, step, False) + removal_values(relation)
                    )
                    counted += 1
            previous = present
            values = self.measures(state)
            queries.append(self.query(step, seq, block, values))
            dates.append(self.snapshots[seq]["game_date"])
            measured.append(values)
            if len(queries) == CHUNK:
                stage.write(self.chunk(index, counter, observations, fact_rows, queries))
                observations, fact_rows, queries = [], {}, []
        if queries:
            stage.write(self.chunk(index, counter, observations, fact_rows, queries))
        for seq in self.moments[len(self.steps) :]:
            measured.append(self.measures(self.current(seq)))
            dates.append(self.snapshots[seq]["game_date"])
        labels = self.labels(dates, measured)
        stage.write(node_tables(index, name, self.kinds, self.keys))
        stage.write(label_tables(index, counter, labels, held_out))
        LOG.info(
            "run build exit run=%d seconds=%.2f steps=%d entities=%d labels=%d",
            self.run["run_id"],
            time.perf_counter() - started,
            len(self.steps),
            len(self.kinds),
            len(labels),
        )
        return dict(steps=len(self.steps), entities=len(self.kinds), labels=len(labels))

    def chunk(self, index, counter, observations, fact_rows, queries):
        return chunk_tables(
            index, counter, self.kinds, self.features, observations, fact_rows, queries, self.player
        )

    def measures(self, state):
        player = (self.run["player_country"],)
        economy = state["country_economy_report"].get(player)
        people = state["player_population"].get(())
        living = state["country_standard_of_living_report"].get(player)
        return (
            logarithm(None if economy is None else economy["gdp"]),
            logarithm(None if people is None else sum(people[name] for name in STRATA)),
            finite(None if living is None else living["average_standard_of_living_by_population"]),
        )

    def query(self, step, seq, block, values):
        names = [row["name"] for row in self.actions[seq]]
        rest = [name for name in names if name != EVENT]
        category = GROUP[rest[0]] if rest else "events" if names else "empty"
        return dict(
            step=step,
            period=self.snapshots[seq]["game_date"] // HOURS_PER_DAY,
            key=seq,
            category=category,
            subject=block,
            x=self.vector("query", {}),
            signal=[0.0 if value is None else value for value in values],
            observed=[float(value is not None) for value in values],
        )

    def labels(self, dates, measured):
        rows = []
        for i in range(len(self.steps)):
            for code, name in enumerate(SERIES):
                base = measured[i][code]
                if base is None:
                    continue
                seen = {dates[i]}
                for j in range(i + 1, len(dates)):
                    tau = (dates[j] - dates[i]) / HOURS_PER_DAY
                    if tau > HORIZON_DAYS:
                        break
                    if dates[j] in seen or measured[j][code] is None:
                        continue
                    seen.add(dates[j])
                    rows.append((i, tau, measured[j][code] - base, name))
        return rows

    def put(self, facts, relation, source, target, *values):
        try:
            facts.setdefault(relation, {})[(source, target)] = tuple(float(value) for value in values)
        except TypeError as error:
            raise ValueError(
                "%s attribute is missing between %s and %s" % (relation, self.keys[source], self.keys[target])
            ) from error

    def history(self, kind, row):
        values, codes = HISTORY[kind]
        return {field(kind, column): row[column] for column in values + codes}

    def reports(self, kind, state, key):
        data = {}
        for table in REPORTS[kind]:
            row = state[table].get(key)
            if row is not None:
                data.update((field(table, column), row[column]) for column in self.columns[table])
        return data

    def subjects(self, kind, state, key):
        data = {}
        for table, columns in SUBJECTS[kind].items():
            row = state[table].get(key)
            if row is not None:
                data.update((field(table, column), row[column]) for column in columns)
        return data

    def world(self, state, features, facts):
        player = self.run["player_country"]
        for (ident,), row in state["country_history"].items():
            node = self.entity("country", ident)
            data = self.history("country", row)
            data.update(self.reports("country", state, (ident,)))
            if ident == player:
                for table in PLAYER:
                    found = state[table].get(())
                    if found is not None:
                        data.update((field(table, column), found[column]) for column in self.columns[table])
                policy = state["player_policy"].get(())
                if policy is not None:
                    data.update((field("player_policy", column), policy[column]) for column in POLICY)
                queue = [
                    row for row in state["player_construction_queue_item"].values() if not row["private"]
                ]
                data.update(
                    queue_work=sum(row["construction_left"] for row in queue),
                    queue_speed=sum(row["speed"] for row in queue),
                    queue_items=len(queue),
                )
            features[node] = data
        for (ident,), row in state["state_history"].items():
            node = self.entity("state", ident)
            data = self.history("state", row)
            data.update(self.reports("state", state, (ident,)))
            data.update(self.subjects("state", state, (ident,)))
            data.update(self.shapes.get(row["state_region"], {}))
            features[node] = data
            if row["country"] is not None:
                self.put(facts, "owns", self.owned("country", row["country"], state), node)
        for (ident,), row in state["player_adjacent_country"].items():
            self.put(facts, "adjacent_country", self.player, self.owned("country", ident, state))
        for (direction, goods), row in state["player_tariff"].items():
            self.put(facts, direction + "_tariff", self.player, self.code("goods", goods), row["level"])
        for (ident,), row in state["market_history"].items():
            node = self.entity("market", ident)
            features[node] = {}
            self.put(facts, "market_owner", node, self.owned("country", row["owner"], state))
        for (market, goods), row in state["market_goods"].items():
            self.put(
                facts,
                "market_goods",
                self.owned("market", market, state),
                self.code("goods", goods),
                row["price"],
                row["buy_orders"],
                row["sell_orders"],
            )
        areas = {area: row["market"] for (area,), row in state["market_area_history"].items()}
        for (area, ident), row in state["market_area_state"].items():
            if area not in areas:
                raise ValueError("market area %r is not present at this snapshot" % area)
            self.put(
                facts,
                "in_market",
                self.owned("state", ident, state),
                self.owned("market", areas[area], state),
                row["port"],
            )
        for (ident, goods), row in state["state_goods"].items():
            source, target = self.owned("state", ident, state), self.code("goods", goods)
            self.put(facts, "local_price", source, target, row["local_price"])
            if row["production"] is not None:
                self.put(facts, "produces_state", source, target, row["production"])
            if row["consumption"] is not None:
                self.put(facts, "consumes_state", source, target, row["consumption"])
        for (ident,), row in state["building_history"].items():
            node = self.entity("building", ident)
            data = self.history("building", row)
            data[field("building", "building_type")] = self.names["building_type"][row["building_type"]]
            data.update(self.subjects("building", state, (ident,)))
            features[node] = data
            if row["state"] is not None:
                self.put(facts, "in_state", node, self.owned("state", row["state"], state))
        for (ident, goods), row in state["building_goods"].items():
            building, target = self.owned("building", ident, state), self.code("goods", goods)
            if row["output"] is not None:
                self.put(facts, "produces", building, target, row["output"])
            if row["input"] is not None:
                self.put(facts, "consumes", target, building, row["input"])
        for (ident, method), row in state["building_production_method_active"].items():
            self.put(
                facts, "uses", self.owned("building", ident, state), self.code("production_method", method)
            )
        groups = {}
        for (private, position), row in sorted(state["player_construction_queue_item"].items()):
            group = (private, row["state"], row["building_type"])
            rank = groups.get(group, 0)
            groups[group] = rank + 1
            node = self.entity(
                "queue_entry", "%d:%d:%d:%d" % (private, row["state"], row["building_type"], rank)
            )
            features[node] = {
                "queue_private": int(private),
                "queue_left": row["construction_left"],
                "queue_rate": row["speed"],
            }
            self.put(facts, "queued", node, self.owned("state", row["state"], state))
            self.put(facts, "will_be", node, self.code("building_type", row["building_type"]))
        for (ident,), row in state["law_history"].items():
            enacting = not row["active"] and row["phase"] in ENACTING_PHASES
            if not row["active"] and not enacting:
                continue
            country = self.owned("country", row["country"], state)
            kind = self.code("law_type", row["law_type"])
            self.put(facts, "has_law" if row["active"] else "enacting", country, kind, row["progress"])
            if enacting:
                node = self.entity("law", ident)
                data = self.history("law", row)
                private = state["law_private"].get((ident,))
                if private is not None:
                    data.update(
                        (field("law", column), private[column]) for column in LAW_PRIVATE[0] + LAW_PRIVATE[1]
                    )
                features[node] = data
                self.put(facts, "pending", country, node)
                self.put(facts, "of_type", node, kind)
        researchers = {}
        for (ident,), row in state["country_research_history"].items():
            researchers[ident] = self.owned("country", row["country"], state)
            if row["researching"] is not None:
                self.put(
                    facts, "researching", researchers[ident], self.code("technology", row["researching"])
                )
        for (ident, technology), row in state["country_research_acquired"].items():
            if ident not in researchers:
                raise ValueError("research row %r has no present country_research" % ident)
            self.put(facts, "researched", researchers[ident], self.code("technology", technology))
        for (ident,), row in state["decree_history"].items():
            if row["state"] is not None:
                self.put(
                    facts,
                    "decreed",
                    self.owned("state", row["state"], state),
                    self.code("decree_type", row["decree_type"]),
                )
        for (ident,), row in state["institution_history"].items():
            self.put(
                facts,
                "institution",
                self.owned("country", row["country"], state),
                self.code("institution_type", row["institution_type"]),
                row["investment"],
            )
        for (ident,), row in state["company_history"].items():
            node = self.entity("company", ident)
            data = self.history("company", row)
            features[node] = data
            if row["building"] is not None:
                self.put(facts, "runs", node, self.owned("building", row["building"], state))

    def layers(self, state, features, facts):
        for (ident,), row in state["building_ownership_history"].items():
            owner = OWNER_KINDS.get(row["owner_kind"])
            if owner is None:
                raise ValueError("unknown building owner kind %r" % row["owner_kind"])
            kind, column = owner
            self.put(
                facts,
                "owned_by",
                self.owned("building", row["building"], state),
                self.owned(kind, row[column], state),
                row["levels"],
            )
        needs = {}
        for (ident, culture, need, goods), row in state["state_pop_need"].items():
            needs[(ident, goods)] = needs.get((ident, goods), 0.0) + row["weight"]
        for (ident, goods), weight in needs.items():
            self.put(facts, "needs", self.owned("state", ident, state), self.code("goods", goods), weight)
        for source_kind, source_name, target_kind, target_name in self.unlocks:
            self.put(
                facts, "unlocks", self.named(source_kind, source_name), self.named(target_kind, target_name)
            )
        self.strata(state, features, facts)
        regions = {ident: row["state_region"] for (ident,), row in state["state_history"].items()}
        by_region = {}
        for ident, region in regions.items():
            by_region.setdefault(region, []).append(ident)
        for ident, region in regions.items():
            for other, border in self.borders.get(region, {}).items():
                for target in by_region.get(other, ()):
                    self.put(
                        facts, "adjacent", self.entity("state", ident), self.entity("state", target), border
                    )
        for (ident, direction, other), row in state["state_migration"].items():
            source, target = (ident, other) if direction == "outbound" else (other, ident)
            self.put(
                facts, "migrates_to", self.owned("state", source, state), self.owned("state", target, state)
            )
        for (ident,), row in state["world_market_hub"].items():
            node = self.entity("world_market_hub", ident)
            features[node] = self.history("world_market_hub", row)
            self.put(facts, "hub", self.owned("state", ident, state), node)
        for (ident,), row in state["shipping_lane_history"].items():
            self.put(
                facts,
                "lane",
                self.owned("state", row["begin_point"], state),
                self.owned("state", row["end_point"], state),
                row["convoy_cost"],
            )
        for (ident,), row in state["strait_endpoint"].items():
            strait = state["strait_history"].get((ident,))
            if strait is None:
                raise ValueError("strait endpoint %r has no present strait" % ident)
            first = self.owned("state", row["first_state"], state)
            second = self.owned("state", row["second_state"], state)
            self.put(facts, "strait", first, second, strait["first_toll_rate"])
            self.put(facts, "strait", second, first, strait["second_toll_rate"])
        self.politics(state, features, facts)

    def strata(self, state, features, facts):
        cultures = {ident: row["culture_type"] for (ident,), row in state["culture_history"].items()}
        types = {ident: row["building_type"] for (ident,), row in state["building_history"].items()}
        groups = {}
        for (ident,), row in state["pop_history"].items():
            status = state["pop_status"].get((ident,))
            if status is None:
                raise ValueError("pop %r has no status at this snapshot" % ident)
            key = (row["location"], row["pop_type"], row["culture"], row["religion"])
            group = groups.setdefault(key, dict(sums={}, means={}, size=0.0, work={}))
            size = float(status["workforce"] + status["dependents"])
            group["size"] += size
            for column in STRATUM_SUMS:
                group["sums"][column] = group["sums"].get(column, 0.0) + status[column]
            for column in STRATUM_MEANS:
                group["means"][column] = group["means"].get(column, 0.0) + status[column] * size
            if row["workplace"] is not None:
                kind = types.get(row["workplace"])
                if kind is None:
                    raise ValueError("pop %r works in a building that is not present" % ident)
                group["work"][kind] = group["work"].get(kind, 0.0) + status["workforce"]
        for (location, pop_type, culture, religion), group in groups.items():
            node = self.entity("stratum", "%d:%d:%d:%d" % (location, pop_type, culture, religion))
            if culture not in cultures:
                raise ValueError("culture %r is not present at this snapshot" % culture)
            data = {field("stratum", column): value for column, value in group["sums"].items()}
            if group["size"] > 0:
                data.update(
                    (field("stratum", column), value / group["size"])
                    for column, value in group["means"].items()
                )
            data[field("stratum", "pop_type")] = self.names["pop_type"][pop_type]
            data[field("stratum", "culture_type")] = self.names["culture_type"][cultures[culture]]
            data[field("stratum", "religion")] = self.names["religion"][religion]
            features[node] = data
            self.put(facts, "lives_in", node, self.owned("state", location, state))
            workforce = group["sums"]["workforce"]
            for kind, count in group["work"].items():
                if workforce > 0:
                    self.put(facts, "works_in", node, self.code("building_type", kind), count / workforce)

    def politics(self, state, features, facts):
        rulers = {
            row["ruler"]: ident
            for (ident,), row in state["country_history"].items()
            if row["ruler"] is not None
        }
        for (ident,), row in state["country_history"].items():
            if row["power_bloc"] is not None:
                self.put(
                    facts,
                    "in_bloc",
                    self.entity("country", ident),
                    self.owned("power_bloc", row["power_bloc"], state),
                )
        for (ident,), row in state["power_bloc_history"].items():
            node = self.entity("power_bloc", ident)
            data = self.history("power_bloc", row)
            data[field("power_bloc", "power_bloc_identity")] = self.names["power_bloc_identity"][
                row["power_bloc_identity"]
            ]
            features[node] = data
        for (bloc, principle), row in state["power_bloc_adopted_principle"].items():
            name = self.names["power_bloc_principle"][principle]
            features[self.owned("power_bloc", bloc, state)]["principle." + name] = 1
        leaders = {}
        for (ident,), row in state["interest_group_history"].items():
            node = self.entity("interest_group", ident)
            data = self.history("interest_group", row)
            data[field("interest_group", "interest_group_type")] = self.names["interest_group_type"][
                row["interest_group_type"]
            ]
            data.update(self.subjects("interest_group", state, (ident,)))
            features[node] = data
            self.put(facts, "represents", node, self.owned("country", row["country"], state))
            if row["leader"] is not None:
                leaders[row["leader"]] = node
        for (ident,), row in state["character_history"].items():
            node = self.entity("character", ident)
            data = self.history("character", row)
            data[field("character", "ideology")] = self.names["ideology"].get(
                row["ideology"], row["ideology"]
            )
            data[field("character", "rank")] = self.names["commander_rank"].get(row["rank"], row["rank"])
            data[field("character", "religion")] = self.names["religion"][row["religion"]]
            data[field("character", "ruler")] = int(ident in rulers)
            features[node] = data
            if ident in rulers:
                self.put(facts, "rules", node, self.entity("country", rulers[ident]))
            if ident in leaders:
                self.put(facts, "leads", node, leaders[ident])
        for (ident, role), row in state["character_assigned_role"].items():
            features[self.owned("character", ident, state)]["role." + self.names["character_role"][role]] = 1
        for (ident,), row in state["political_movement_history"].items():
            node = self.entity("political_movement", ident)
            data = self.history("political_movement", row)
            data[field("political_movement", "political_movement_type")] = self.names[
                "political_movement_type"
            ][row["political_movement_type"]]
            data[field("political_movement", "ideology")] = self.names["ideology"].get(
                row["ideology"], row["ideology"]
            )
            features[node] = data
            self.put(facts, "movement_of", node, self.owned("country", row["country"], state))
        for (group, movement), row in state["interest_group_movement_support"].items():
            self.put(
                facts,
                "supports",
                self.owned("political_movement", movement, state),
                self.owned("interest_group", group, state),
                row["clout"],
            )
        for (ident,), row in state["war_history"].items():
            features[self.entity("war", ident)] = self.history("war", row)
        for (war, country), row in state["war_participant"].items():
            self.put(
                facts,
                "in_war",
                self.owned("country", country, state),
                self.owned("war", war, state),
                row["war_support"],
            )
        occupied = {ident: row["state"] for (ident,), row in state["occupation_history"].items()}
        for (occupation, country), row in state["occupation_country"].items():
            if occupation not in occupied:
                raise ValueError("occupation %r is not present at this snapshot" % occupation)
            self.put(
                facts,
                "occupies",
                self.owned("country", country, state),
                self.owned("state", occupied[occupation], state),
                row["fraction"],
            )
        thinkers = {ident: row["country"] for (ident,), row in state["country_ai_history"].items()}
        for (ai, target), row in state["country_ai_goal"].items():
            if ai not in thinkers:
                raise ValueError("country_ai %r is not present at this snapshot" % ai)
            self.put(
                facts,
                "attitude_%d" % row["attitude"],
                self.owned("country", thinkers[ai], state),
                self.owned("country", target, state),
                row["neighbour"],
                row["diplomatic_relevance"],
            )
        for (ai, region), row in state["country_ai_region_stance"].items():
            if ai not in thinkers:
                raise ValueError("country_ai %r is not present at this snapshot" % ai)
            self.put(
                facts,
                self.names["ai_region_stance"][row["stance"]],
                self.owned("country", thinkers[ai], state),
                self.named("strategic_region", self.names["strategic_region"][region]),
            )

    def owned(self, kind, ident, state):
        if (ident,) not in state[kind + "_history"]:
            raise ValueError("%s %r is referenced but not present at this snapshot" % (kind, ident))
        return self.entity(kind, ident)

    def actions_at(self, step, seq, features, facts):
        state = {table: versions.current for table, versions in self.tables.items()}
        actions = self.actions[seq]
        block = self.entity("block", seq)
        features[block] = dict(
            turn=step + 1,
            actions=len(actions),
            events=sum(row["name"] == EVENT for row in actions),
            days=(self.snapshots[seq]["game_date"] - self.start) / HOURS_PER_DAY,
        )
        for position, row in enumerate(actions):
            node = self.entity("action", row["n"])
            data = dict(name=row["name"], group=GROUP[row["name"]], advisor=row["advisor"], position=position)
            self.put(facts, "in_block", node, block)
            for name, value in row["params"].items():
                if isinstance(value, str):
                    data["param." + name] = value
                    if name in TEXT_TARGETS:
                        self.put(facts, "targets", node, self.named(TEXT_TARGETS[name], value))
                elif name in ENTITY_TARGETS:
                    self.put(facts, "targets", node, self.owned(ENTITY_TARGETS[name], value, state))
                elif name in RESOLVED_TARGETS:
                    kind = RESOLVED_TARGETS[name]
                    found = state[name + "_history"].get((value,))
                    if found is None:
                        raise ValueError("%s %r is named by an action but not present" % (name, value))
                    self.put(facts, "targets", node, self.code(kind, found[kind]))
                elif name in CHOICE_PARAMS:
                    data["param." + name] = value
                elif name not in UNMODELLED_PARAMS:
                    raise ValueError("undeclared action parameter %s" % name)
            features[node] = data
        return block


def geometry(world, terrains):
    regions = {}
    for row in world["provinces"]:
        region = row["state_region"]
        if region is None:
            continue
        shape = regions.setdefault(region, dict(area=0.0, x=0.0, y=0.0, provinces=0, coastal=0, terrain={}))
        area = float(row["area_pixels"])
        shape["area"] += area
        shape["x"] += row["x"] * area
        shape["y"] += row["y"] * area
        shape["provinces"] += 1
        shape["terrain"][row["terrain"]] = shape["terrain"].get(row["terrain"], 0.0) + area
    region_of = {row["color"]: row["state_region"] for row in world["provinces"]}
    borders = {}
    for row in world["adjacency"]:
        first, second = region_of[row["first_color"]], region_of[row["second_color"]]
        if first is None or second is None:
            for region in (first, second):
                if region is not None:
                    regions[region]["coastal"] = 1
            continue
        if first == second:
            continue
        for source, target in ((first, second), (second, first)):
            found = borders.setdefault(source, {})
            found[target] = found.get(target, 0.0) + row["shared_border_pixels"]
    shapes = {}
    for region, shape in regions.items():
        data = {
            field("state", "x"): shape["x"] / shape["area"],
            field("state", "y"): shape["y"] / shape["area"],
            field("state", "area"): shape["area"],
            field("state", "provinces"): shape["provinces"],
            field("state", "coastal"): shape["coastal"],
        }
        data.update(
            ("terrain." + terrains[code], area / shape["area"]) for code, area in shape["terrain"].items()
        )
        shapes[region] = data
    return shapes, borders


def allowed_fields(columns, parameters, names):
    numbers, categories = set(NUMBERS), set(CATEGORIES)
    numbers.update("terrain." + name for name in names["terrain"].values())
    numbers.update("role." + name for name in names["character_role"].values())
    numbers.update("principle." + name for name in names["power_bloc_principle"].values())
    numbers.update(field(table, column) for table, names in columns.items() for column in names)
    for name, kinds in parameters.items():
        if "text" in kinds:
            categories.add("param." + name)
        if "integer" in kinds and name in CHOICE_PARAMS:
            numbers.add("param." + name)
    return sorted(numbers), sorted(categories)
