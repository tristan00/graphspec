from datetime import date

import numpy as np
import pyarrow as pa
import pytest

from churn.schema import BY_FIELD
from churn.truth import truth
from graphspec.infra.features import encoder
from totalwar import corpus
from totalwar.build import Counter
from totalwar.domain import CATEGORIES, NUMBERS, PARTS
from totalwar.graph import Campaign
from totalwar.reference import FAMILIES, Reference
from totalwar.reference_tables import TABLES, keyed
from totalwar.subjects import ActionSubject


def campaign():
    reference = Reference.__new__(Reference)
    for table in TABLES:
        setattr(reference, table.name, {})
    reference.keys = {family: {} for family in FAMILIES}
    reference.keys["faction"][1] = "player"
    reference.ids = {
        family: {key: identifier for identifier, key in values.items()}
        for family, values in reference.keys.items()
    }
    reference.enum, reference.actions, reference.scopes, reference.skill_sets = {}, {}, {}, {}
    rows = {name: [] for name in corpus.TABLES}
    rows.update({name: [] for name, *_ in corpus.SETS})
    rows["snapshots"] = [
        dict(snapshot_id=11, kind_id=1, turn=1, ts=1, faction_id=1, **{part: 1 for part in PARTS})
    ]
    return Campaign(
        reference,
        encoder(8, list(NUMBERS), list(CATEGORIES)),
        16,
        dict(campaign_id=1, campaign_key="small"),
        rows,
    )


def test_totalwar_minimal_campaign_uses_shared_tables_without_null_nodes():
    graph = campaign()
    tables = graph.build(Counter(0), 0, "small")
    assert graph.keys == ["player"]
    assert tables[("query", None, None)]["category"].to_pylist() == ["no_action"]
    assert tables[("reward", None, None)]["y"].to_pylist() == [0.0]
    assert tables[("query", None, None)]["signal_observed"][0].as_py() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert graph.step_at(2) == len(graph.snapshots)
    assert all(key is not None for _, key in graph.entities)
    with pytest.raises(ValueError, match="non-null"):
        graph.entity("skill", None)


def test_totalwar_action_subjects_reject_unknown_types_and_preserve_node_zero():
    graph = campaign()
    actor = graph.faction("player")
    data = {}
    subject = ActionSubject(graph, actor, {}, {}, data)
    assert subject.resolve("noop", "") == 0
    assert subject.resolve("move", "xy:2.5,3") == actor
    assert data == {"x": 2.5, "y": 3.0}
    with pytest.raises(ValueError, match="undeclared"):
        subject.resolve("unrecognized", "")


def test_totalwar_reference_selection_never_uses_another_factions_skill_set():
    graph = campaign()
    graph.ref.skill_sets = {"wizard": {"other-faction": "wrong"}}
    assert graph.ref.skill_set("wizard", "player") is None
    graph.ref.skill_sets["wizard"][""] = "shared"
    assert graph.ref.skill_set("wizard", "player") == "shared"
    graph.ref.skill_sets["wizard"]["player"] = "specific"
    assert graph.ref.skill_set("wizard", "player") == "specific"
    graph.ref.tech_sets = {
        "one": dict(key="one", faction_key="player", culture="", subculture=""),
        "two": dict(key="two", faction_key="player", culture="", subculture=""),
    }
    with pytest.raises(ValueError, match="ambiguous"):
        graph.ref.tech_set("player")
    with pytest.raises(ValueError, match="duplicate"):
        keyed([dict(key="same"), dict(key="same")], "key")


def test_totalwar_catalogue_does_not_create_a_skill_for_missing_background():
    from totalwar.catalogue import catalogue

    graph = campaign()
    graph.rows["extensions"] = [dict(background_skill_id=None)]
    graph.rows["candidates"] = [dict(bg_skill_id=None, cand_subtype_id=None, trait_ids=None)]
    catalogue(graph)
    assert not graph.entities


def test_totalwar_events_after_last_snapshot_cannot_be_applied_to_it():
    from totalwar.events import consequences

    graph = campaign()
    graph.rows["events"] = [dict(ts=2)]
    assert consequences(graph) == {}
    assert graph.keys == ["player"]


def test_totalwar_campaign_state_applies_mission_payloads():
    from totalwar.state import campaign_state

    graph = campaign()
    graph.ref.keys["mission"] = {1: "mission"}
    graph.ref.mission_payloads = {
        "mission": [
            dict(payload_key="TREASURY", value="AMOUNT[20]", status_key="SUCCESS"),
            dict(payload_key="ADD_ANCILLARY", value="ANCILLARY_KEY[sword]", status_key="SUCCESS"),
        ]
    }
    row = dict(
        mission_id=1,
        issuer_id=None,
        category="mission",
        status_id=1,
        turns_remaining=2,
        is_quest=False,
        is_victory=False,
        completed=False,
        cancelled=False,
        pending=True,
    )
    features, facts = {}, []
    campaign_state(
        graph,
        lambda table: [row] if table == "missions" else [],
        features,
        lambda *args, **kwargs: facts.append((args, kwargs)),
    )
    mission = graph.entities[("mission", "mission")]
    assert features[mission]["treasury"] == 20
    assert facts[-1] == (
        ("mission_payload", mission, graph.entities[("item", "sword")], None, True),
        {"tag": ("item", "sword")},
    )


def test_churn_entity_schema_controls_namespace_and_hierarchy():
    city = BY_FIELD["client_city"]
    assert city.keys(
        {"client_city": [" Austin ", "Austin", ""], "client_state_code": ["TX", "MA", "TX"]}
    ) == ["TX:austin", "MA:austin", ""]
    assert BY_FIELD["naics_code"].members(["541110", "null"]) == ["54", "541", "5411", "54111", "541110"]


def test_churn_city_edges_and_cutoff_history_are_independent_of_field_order():
    row = dict(
        policy_key="p",
        client_key="client",
        pol_policy_effective_date="2023-01-01",
        pol_no_longer_in_force_date="2024-01-01",
        pol_expiration_date="2025-01-01",
        pol_policy_cancelled_flag="unavailable",
        client_city="Austin",
        client_state_code="TX",
    )
    graph = truth(pa.Table.from_pylist([row]), date(2024, 1, 1), ("client_city", "client_state_code"))
    assert graph.tables[("edge_city_state", "client_city", "client_state")].num_rows == 1
    row["pol_no_longer_in_force_date"] = "2023-12-31"
    with pytest.raises(ValueError, match="cancellation"):
        truth(pa.Table.from_pylist([row]), date(2024, 1, 1), ())


def test_totalwar_verifier_reads_staged_edges_without_a_graph_database(tmp_path):
    from graphspec.infra.staging import Stage
    from totalwar.verify import StagedChecks

    graph = campaign()
    tables = graph.build(Counter(2 << 32), 2, "small")
    player = tables[("faction", None, None)]["id"][0].as_py()
    tables[("edge_standing", "faction", "faction")] = pa.table(
        {
            "from": np.array([player, player], dtype=np.int64),
            "to": np.array([player, player], dtype=np.int64),
            "step": np.array([0, 1], dtype=np.int32),
            "present": [True, False],
        }
    )
    with Stage(tmp_path / "stage") as stage:
        stage.write(tables)
    source = StagedChecks(tmp_path / "stage")
    assert source.count("standing", 2, 0) == 1
    assert source.count("standing", 1, 0) == 0
    assert source.count("standing", 2, 1) == 0
    assert source.labels[source.queries[0]["id"]]["y"] == 0.0
