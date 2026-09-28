from types import SimpleNamespace

import numpy as np
import pytest

from graphspec.infra.features import encoder
from graphspec.infra.staging import Stage
from graphspec.staged import Staged
from vic3.build import held
from vic3.domain import HOURS_PER_DAY, PLAYER, REPORTS, TARGET, VOCABULARIES
from vic3.graph import Run, allowed_fields
from vic3.source import KEYS, Versions
from vic3.tables import Counter
from vic3.verify import check_graph, check_labels

DAY = 24
COLUMNS = {table: [] for table in REPORTS["country"] + REPORTS["state"] + PLAYER}
COLUMNS["country_economy_report"] = ["gdp"]
COLUMNS["player_population"] = ["lower_strata", "middle_strata", "upper_strata"]
COLUMNS["country_standard_of_living_report"] = ["average_standard_of_living_by_population"]
NAMES = {name: {} for name in VOCABULARIES}
NAMES.update(
    goods={1: "grain"},
    building_type={3: "building_farm"},
    law_type={2: "law_x", 4: "law_y"},
    technology={9: "tech_a"},
    terrain={1: "plains", 16: "ocean"},
    pop_type={8: "farmers"},
    culture_type={5: "cult"},
    religion={3: "rel"},
    ideology={1: "ideology_a"},
    interest_group_trait={1: "trait_a"},
    interest_group_type={1: "ig_x"},
)
WORLD = dict(
    provinces=[
        dict(color=1, state_region=10, terrain=1, area_pixels=100, x=0.0, y=0.0),
        dict(color=2, state_region=11, terrain=1, area_pixels=300, x=4.0, y=0.0),
        dict(color=3, state_region=None, terrain=16, area_pixels=50, x=2.0, y=2.0),
    ],
    adjacency=[
        dict(first_color=1, second_color=2, shared_border_pixels=7),
        dict(first_color=2, second_color=3, shared_border_pixels=3),
    ],
)
DEFINITIONS = dict(
    unlocks=[("technology", "tech_a", "building_type", "building_farm")],
    disallows=[("law_type", "law_x", "law_type", "law_y"), ("law_type", "law_y", "production_method", "pm_x")],
    stances={"ideology_a": [("law_x", 2.0), ("law_y", -1.0)], "ideology_b": [("law_x", -2.0)]},
    laws={
        "law_x": {"law_type.group": "lawgroup_a", "modifier.state_x_mult": 0.5},
        "law_y": {"law_type.group": "lawgroup_a", "modifier.state_y_bool": 1.0},
    },
)
MODIFIERS = ["state_x_mult", "state_y_bool"]
PARAMETERS = {
    "building_type": {"text"},
    "state": {"integer"},
    "event": {"text"},
    "option": {"integer"},
    "days": {"integer"},
    "building": {"integer"},
    "production_method": {"text"},
    "technology": {"text"},
}


def versions(rows):
    return {table: Versions(table, keys, rows.get(table, [])) for table, keys in KEYS.items()}


def data():
    rows = {
        "country_history": [
            dict(
                country=7,
                seq=0,
                present=True,
                in_civil_war=False,
                bloc_leader=False,
                government=1,
                country_rank=2,
                country_type=1,
                religion=3,
                ruler=None,
                power_bloc=None,
            ),
            dict(
                country=8,
                seq=0,
                present=True,
                in_civil_war=False,
                bloc_leader=False,
                government=1,
                country_rank=2,
                country_type=1,
                religion=3,
                ruler=None,
                power_bloc=None,
            ),
        ],
        "state_history": [
            dict(state=1, seq=0, present=True, country=7, state_region=10, treaty_port_potential=False),
            dict(state=2, seq=0, present=True, country=8, state_region=11, treaty_port_potential=False),
            dict(state=2, seq=4, present=True, country=7, state_region=11, treaty_port_potential=False),
        ],
        "country_economy_report": [
            dict(country=7, seq=0, present=True, gdp=100.0),
            dict(country=7, seq=1, present=True, gdp=999.0),
            dict(country=7, seq=3, present=True, gdp=110.0),
            dict(country=7, seq=4, present=True, gdp=121.0),
            dict(country=7, seq=6, present=True, gdp=133.1),
        ],
        "player_population": [
            dict(seq=0, lower_strata=600, middle_strata=300, upper_strata=100),
            dict(seq=6, lower_strata=900, middle_strata=450, upper_strata=150),
        ],
        "country_standard_of_living_report": [
            dict(country=7, seq=0, present=True, average_standard_of_living_by_population=10.0),
            dict(country=7, seq=3, present=True, average_standard_of_living_by_population=12.0),
            dict(country=7, seq=6, present=True, average_standard_of_living_by_population=9.0),
        ],
        "building_history": [
            dict(
                building=5,
                seq=0,
                present=True,
                building_type=3,
                state=1,
                levels=2,
                active=True,
                subsidized=False,
                dead=False,
            ),
            dict(
                building=5,
                seq=4,
                present=False,
                building_type=None,
                state=None,
                levels=None,
                active=None,
                subsidized=None,
                dead=None,
            ),
        ],
        "building_goods": [
            dict(building=5, goods=1, seq=0, present=True, input=None, output=3.0, slave_input=None),
            dict(building=5, goods=1, seq=4, present=False, input=None, output=None, slave_input=None),
        ],
        "state_goods": [
            dict(
                state=1,
                goods=1,
                seq=0,
                present=True,
                local_price=2.0,
                production=1.0,
                consumption=None,
                trade=None,
            )
        ],
        "market_history": [dict(market=0, seq=0, present=True, owner=7)],
        "market_area_history": [dict(market_area=0, seq=0, present=True, market=0)],
        "market_area_state": [dict(market_area=0, state=1, seq=0, present=True, port=False)],
        "player_tariff": [
            dict(direction="import", goods=1, seq=0, present=True, level=4, previous_level=3, last_change=0)
        ],
        "player_construction_queue_item": [
            dict(
                private=False,
                position=0,
                seq=0,
                present=True,
                state=1,
                building_type=3,
                owner_building_type=None,
                speed=1.0,
                construction_left=10.0,
            )
        ],
        "law_history": [
            dict(
                law=1,
                seq=0,
                present=True,
                country=7,
                law_type=2,
                progress=0.0,
                phase=0,
                checkpoint_or_stop_date=None,
                active=True,
                imposer=None,
            ),
            dict(
                law=2,
                seq=0,
                present=True,
                country=7,
                law_type=4,
                progress=0.4,
                phase=1,
                checkpoint_or_stop_date=5,
                active=False,
                imposer=None,
            ),
            dict(
                law=3,
                seq=0,
                present=True,
                country=8,
                law_type=4,
                progress=0.0,
                phase=0,
                checkpoint_or_stop_date=None,
                active=False,
                imposer=None,
            ),
        ],
        "law_private": [
            dict(
                law=2,
                seq=0,
                present=True,
                checkpoint_result=2,
                replacing_law_type=None,
                enactment_start_date=1,
                chance=0.6,
            )
        ],
        "building_ownership_history": [
            dict(
                building_ownership=1,
                seq=0,
                present=True,
                building=5,
                owner_kind="country",
                owner_country=7,
                owner_building=None,
                owner_company=None,
                levels=2,
                privatization=False,
                in_owner_assets=False,
            ),
            dict(
                building_ownership=1,
                seq=4,
                present=False,
                building=None,
                owner_kind=None,
                owner_country=None,
                owner_building=None,
                owner_company=None,
                levels=None,
                privatization=None,
                in_owner_assets=None,
            ),
        ],
        "state_pop_need": [
            dict(state=1, culture_type=5, need=1, goods=1, seq=0, present=True, weight=0.5),
            dict(state=1, culture_type=6, need=1, goods=1, seq=0, present=True, weight=0.25),
        ],
        "culture_history": [dict(culture=20, seq=0, present=True, culture_type=5)],
        "pop_history": [
            dict(
                pop=100,
                seq=0,
                present=True,
                location=1,
                workplace=5,
                culture=20,
                religion=3,
                pop_type=8,
                social_class=1,
                acceptance_status=1,
                assimilation_culture=None,
                conversion_religion=None,
            ),
            dict(
                pop=101,
                seq=0,
                present=True,
                location=1,
                workplace=None,
                culture=20,
                religion=3,
                pop_type=8,
                social_class=1,
                acceptance_status=1,
                assimilation_culture=None,
                conversion_religion=None,
            ),
            dict(
                pop=100,
                seq=4,
                present=False,
                location=None,
                workplace=None,
                culture=None,
                religion=None,
                pop_type=None,
                social_class=None,
                acceptance_status=None,
                assimilation_culture=None,
                conversion_religion=None,
            ),
        ],
        "pop_status": [
            dict(
                pop=100,
                seq=0,
                wealth_progress=0.1,
                num_literate=5,
                loyalists_and_radicals=1,
                wealth=10,
                food_security=1.0,
                political_strength=3,
                workforce=30,
                dependents=10,
                job_satisfaction=0.8,
            ),
            dict(
                pop=101,
                seq=0,
                wealth_progress=0.1,
                num_literate=1,
                loyalists_and_radicals=0,
                wealth=6,
                food_security=0.5,
                political_strength=1,
                workforce=10,
                dependents=10,
                job_satisfaction=0.4,
            ),
        ],
    }
    snapshots = [
        dict(seq=0, phase="start", game_date=100 * DAY),
        dict(seq=1, phase="post_action", game_date=100 * DAY),
        dict(seq=2, phase="post_action", game_date=100 * DAY),
        dict(seq=3, phase="post_run", game_date=130 * DAY),
        dict(seq=4, phase="post_run", game_date=160 * DAY),
        dict(seq=5, phase="post_action", game_date=160 * DAY),
        dict(seq=6, phase="post_run", game_date=190 * DAY),
    ]
    actions = [
        dict(
            n=1,
            turn=1,
            before_seq=0,
            name="choose_event_option",
            advisor="random",
            params={"event": "x.1", "option": 1},
        ),
        dict(
            n=2,
            turn=1,
            before_seq=1,
            name="build",
            advisor="random",
            params={"building_type": "building_farm", "state": 1},
        ),
        dict(
            n=3,
            turn=1,
            before_seq=1,
            name="switch_production_method",
            advisor="random",
            params={"building": 5, "production_method": "pm_x"},
        ),
        dict(n=4, turn=1, before_seq=2, name="run_days", advisor="loop", params={"days": 30}),
        dict(n=5, turn=2, before_seq=3, name="run_days", advisor="random", params={"days": 30}),
        dict(n=6, turn=3, before_seq=4, name="research", advisor="random", params={"technology": "t1"}),
        dict(n=7, turn=3, before_seq=5, name="run_days", advisor="loop", params={"days": 30}),
    ]
    rows.update(
        interest_group_history=[
            dict(
                interest_group=50,
                seq=0,
                present=True,
                country=7,
                interest_group_type=1,
                approval_state=0,
                influence_type=1,
                in_government=True,
                leader=None,
            )
        ],
        interest_group_ideology=[dict(interest_group=50, ideology=1, seq=0, present=True)],
        interest_group_held_trait=[dict(interest_group=50, interest_group_trait=1, seq=0, present=True, enabled=True)],
        interest_group_pop=[dict(interest_group=50, pop=100, seq=0, present=True)],
        party_history=[dict(party=60, seq=0, present=True, country=7, party_type=2, votes=0.4)],
        party_member=[dict(party=60, interest_group=50, seq=0, present=True)],
    )
    return dict(snapshots=snapshots, actions=actions, tables=versions(rows), world=WORLD, definitions=DEFINITIONS)


def make_run(features=16):
    numbers, categories = allowed_fields(COLUMNS, PARAMETERS, NAMES, MODIFIERS)
    encode = encoder(features // 2, numbers, categories)
    run = dict(run_id=2, kind="mini", player_country=7, end_reason="time")
    return Run(encode, features, run, data(), NAMES, COLUMNS, PARAMETERS, frozenset(numbers + categories))


def test_run_writes_a_contract_graph(tmp_path):
    graph = make_run()
    read, advance = [], graph.current
    graph.current = lambda seq: read.append(seq) or advance(seq)
    staging = tmp_path / "vic3.staging"
    with Stage(staging) as stage:
        counts = graph.build(stage, Counter(0), 0, "run-2", held(2))
    assert counts == dict(steps=3, entities=len(graph.kinds), labels=18, absent={})
    assert read == [0, 3, 4, 6]
    staged = Staged(staging)
    reader = staged.reader(TARGET)
    tables = reader.graph(0)
    problems = check_graph(tables, 0)
    check_labels(staged, problems)
    staged.close()
    assert problems == []
    queries = tables["queries"]
    assert queries.column("category").to_pylist() == ["construction", "empty", "research"]
    assert queries.column("period").to_pylist() == [100, 130, 160]
    assert queries.column("key").to_pylist() == [0, 3, 4]
    signal = np.asarray(queries.column("signal").combine_chunks().flatten().to_numpy(zero_copy_only=False))
    expected = [[np.log(100.0), np.log(1000.0), 10.0], [np.log(110.0), np.log(1000.0), 12.0]]
    expected.append([np.log(121.0), np.log(1000.0), 12.0])
    assert np.allclose(signal.reshape(3, 3), expected, atol=1e-3)
    nodes = tables["nodes"]
    key_of = dict(zip(nodes.column("entity").to_pylist(), nodes.column("key").to_pylist()))
    subjects = queries.column("subject").to_pylist()
    assert [key_of[node] for node in subjects] == ["block:0", "block:3", "block:4"]
    for role in ("actor", "context"):
        assert [key_of[node] for node in queries.column(role).to_pylist()] == ["country:7"] * 3
    assert "orders" not in tables["edges"]
    in_block = tables["edges"]["in_block"]
    joined = {
        (s, st): t
        for s, t, st, p in zip(
            in_block.column("source").to_pylist(),
            in_block.column("target").to_pylist(),
            in_block.column("step").to_pylist(),
            in_block.column("present").to_pylist(),
        )
        if p
    }
    observations = tables["observations"]
    actions = {
        (node, st)
        for node, st in zip(
            observations.column("entity").to_pylist(), observations.column("step").to_pylist()
        )
        if key_of[node].startswith("action:")
    }
    assert actions == set(joined) and len(actions) == 5
    assert {key_of[node] for node, st in actions if st == 1} == {"action:5"}
    assert all(joined[(node, st)] == subjects[st] for node, st in actions)
    assert {key_of[node] for node, st in actions if st == 0} == {"action:1", "action:2", "action:3"}
    (unit,) = Staged(staging).by_table[TARGET]
    labelled = unit.read()
    series = np.asarray(labelled.column("series").to_pylist())
    targets = tables["targets"]
    steps, tau, y = (np.asarray(targets.column(name).to_pylist()) for name in ("step", "tau", "y"))
    order = np.flatnonzero(series == "gdp")
    assert steps[order].tolist() == [0, 0, 0, 1, 1, 2]
    assert tau[order].tolist() == [30.0, 60.0, 90.0, 30.0, 60.0, 30.0]
    assert np.allclose(y[order], np.log([1.1, 1.21, 1.331, 1.1, 1.21, 1.1]))
    order = np.flatnonzero(series == "population")
    assert np.allclose(y[order], np.log([1.0, 1.0, 1.5, 1.0, 1.5, 1.5]))
    order = np.flatnonzero(series == "sol")
    assert np.allclose(y[order], [2.0, 2.0, -1.0, 0.0, -3.0, -3.0])
    assert tau[order].tolist() == [30.0, 60.0, 90.0, 30.0, 60.0, 30.0]
    owns = tables["edges"]["owns"]
    removed = [
        (s, t)
        for s, t, p in zip(
            owns.column("step").to_pylist(),
            owns.column("target").to_pylist(),
            owns.column("present").to_pylist(),
        )
        if not p
    ]
    assert len(removed) == 1 and removed[0][0] == 2
    in_state = tables["edges"]["in_state"]
    assert in_state.column("present").to_pylist().count(False) == 1
    targets_edges = tables["edges"]["targets"]
    assert targets_edges.column("present").to_pylist().count(True) == 5
    relations = {name: table for name, table in tables["edges"].items() if table.num_rows}
    assert relations["owned_by"].column("present").to_pylist() == [True, True, False]
    assert relations["has_law"].num_rows == 3 and relations["enacting"].num_rows == 3
    assert relations["pending"].num_rows == 3 and relations["of_type"].num_rows == 3
    assert np.allclose(relations["needs"].column("weight").to_numpy(), [0.75] * 3)
    assert relations["unlocks"].num_rows == 3
    assert relations["disallows"].num_rows == 6
    assert relations["holds"].num_rows == relations["governs"].num_rows == 3
    assert relations["party_of"].num_rows == relations["member_of"].num_rows == 3
    assert np.allclose(relations["stance"].column("approval").to_numpy(), [2.0, -1.0] * 3)
    assert relations["backs"].column("present").to_pylist() == [True, True, False]
    assert np.allclose(relations["backs"].column("members").to_numpy()[:2], [40.0, 40.0])
    law = [node for node, key in key_of.items() if key == "law_type:law_x"]
    assert len(law) == 1 and "production_method:pm_x" in key_of.values()
    assert "ideology:ideology_b" not in key_of.values()
    assert relations["adjacent"].num_rows == 6
    assert np.allclose(relations["adjacent"].column("border").to_numpy(), [7.0] * 6)
    assert relations["lives_in"].column("present").to_pylist() == [True, True, True]
    works = relations["works_in"]
    assert works.column("present").to_pylist() == [True, True, False]
    assert np.allclose(works.column("share").to_numpy()[:2], [0.75, 0.75])
    kinds = tables["nodes"].column("kind").to_pylist()
    assert kinds.count("stratum") == 1 and kinds.count("law") == 1


def test_action_targets_absent_at_the_step_are_counted_not_stated():
    graph = make_run()
    graph.actions[0][2]["params"]["building"] = 999
    facts = {}
    graph.current(0)
    graph.actions_at(0, 0, {}, facts)
    assert graph.absent == {"action_target_building": 1}
    assert sorted(graph.kinds[target] for _, target in facts["targets"]) == [
        "building_type",
        "production_method",
        "state",
    ]


def test_unknown_action_parameter_is_an_error():
    graph = make_run()
    graph.actions[0][0]["params"]["mystery"] = 3
    graph.current(0)
    with pytest.raises(ValueError, match="undeclared action parameter"):
        graph.actions_at(0, 0, {}, {})


def test_missing_edge_attribute_is_an_error():
    graph = make_run()
    graph.tables["state_goods"].items[0]["local_price"] = None
    with pytest.raises(ValueError, match="local_price attribute is missing"):
        graph.world(graph.current(0), {}, {})


def test_split_is_deterministic():
    assert held(2) == held(2)
    assert isinstance(held(1), bool)


def test_labels_stop_at_the_horizon_and_skip_unmeasured_points_per_series():
    dates = [d * HOURS_PER_DAY for d in (0, 400, 1825, 1826, 2400)]
    measured = [(0.0, None, 5.0), (0.1, 1.0, 6.0), (0.2, 1.5, None), (0.3, 2.0, 7.0), (0.4, 2.5, 8.0)]
    rows = Run.labels(SimpleNamespace(steps=[0, 1]), dates, measured)
    assert [(i, tau, name) for i, tau, _, name in rows] == [
        (0, 400.0, "gdp"),
        (0, 1825.0, "gdp"),
        (0, 400.0, "sol"),
        (1, 1425.0, "gdp"),
        (1, 1426.0, "gdp"),
        (1, 1425.0, "population"),
        (1, 1426.0, "population"),
        (1, 1426.0, "sol"),
    ]
    assert rows[-1][2] == pytest.approx(1.0)


def test_law_modifiers_parse_into_numbers_and_reject_other_shapes():
    from vic3.source import modifiers

    parsed = modifiers("state_x_mult = 0.25 country_y_bool = yes country_z_bool = no")
    assert parsed == {"state_x_mult": 0.25, "country_y_bool": 1.0, "country_z_bool": 0.0}
    with pytest.raises(ValueError, match="key = value"):
        modifiers("state_x_mult 0.25")
