import pyarrow as pa
import pytest
from test_mappings import DAY, PLAN, UTILITY, mapped, table

from graphspec import SCRIPTED, SPLITS, compute, decisions, evaluate, nodes, scripted, tasks

pytestmark = pytest.mark.builder
ROWS = [dict(id="p0", day=1, start=1, region="n", part="train", world="g1", gap="a", late=True),
        dict(id="p1", day=1, start=1, region="n", part="train", world="g2", gap="b", late=False),
        dict(id="p2", day=2, start=2, region="s", part="score", world="g1", gap=None, late=False),
        dict(id="p3", day=3, start=2, region="s", part="train", world="g2", gap="d", late=True),
        dict(id="p4", day=4, start=4, region="e", part="score", world="g1", gap="e", late=False),
        dict(id="p5", day=5, start=4, region="e", part="train", world="g2", gap="f", late=True),
        dict(id="p6", day=6, start=6, region="w", part="score", world="g1", gap="g", late=False),
        dict(id="p7", day=9, start=6, region="w", part="train", world="g2", gap="h", late=True)]
COLUMNS = dict(id="identifier", day=DAY, start=DAY, region="identifier", part="identifier", world="identifier",
               gap=dict(type="identifier", missing="omit"), late=dict(type="flag", outcome="day + 1"))
HALVES = dict(train=0.5, test=0.5)
FORMS = (
    (dict(name="hash", key="id", parts=HALVES, salt="people"),
     dict(p0="train", p1="test", p2="train", p3="test", p4="test", p5="train", p6="train", p7="train")),
    (dict(name="hash", key=["region", "id"], parts=HALVES, salt="people"),
     dict(p0="train", p1="test", p2="test", p3="test", p4="test", p5="train", p6="train", p7="train")),
    (dict(name="group", by="region", parts=HALVES, salt="regions"),
     dict(p0="train", p1="train", p2="test", p3="test", p4="test", p5="test", p6="train", p7="train")),
    (dict(name="graph", parts=HALVES, salt="worlds"),
     dict(p0="test", p1="train", p2="test", p3="train", p4="test", p5="train", p6="test", p7="train")),
    (dict(name="time", parts=dict(train=0.5, validation=0.25, test=0.25)),
     dict(p0="train", p1="train", p2="train", p3="train", p4="validation", p5="validation", p6="test", p7="test")),
    (dict(name="time", parts=dict(train=0.1, test=0.9)),
     dict(p0="train", p1="train", p2="test", p3="test", p4="test", p5="test", p6="test", p7="test")),
    (dict(name="time", parts=dict(train=None, validation=3, test="span(2, 'day')")),
     dict(p0="train", p1="train", p2="train", p3="train", p4="train", p5="validation", p6="validation", p7="test")),
    (dict(name="time", by="start", parts=HALVES),
     dict(p0="train", p1="train", p2="train", p3="train", p4="test", p5="test", p6="test", p7="test")),
    ("part",
     dict(p0="train", p1="train", p2="score", p3="train", p4="score", p5="train", p6="score", p7="train")),
    ("'train' if region in ['n', 's'] else 'score'",
     dict(p0="train", p1="train", p2="train", p3="train", p4="score", p5="score", p6="score", p7="score")),
)


def declared(folder, split, rows=ROWS, name="rows"):
    source = table(folder, name, rows, COLUMNS)
    thing = nodes(source, key="id", graph="world", holds="always", known=None)
    target = dict(kind="number", series="s", value="1", known="day + 1", about=None, utility=UTILITY)
    return dict(rows=source, flagged=compute(source, columns=dict(flag="not late")), thing=thing,
                ask=tasks(source, key="id", at="day", graph="world", split=split, target=target,
                          references=dict(at=dict(sort="node", of=thing, key="id", absent="stop"))))


def members(folder, split, rows=ROWS):
    found = mapped(folder, declared(folder, split, rows))
    return {at: name for graph in found["graphs"]
            for at, name in zip(graph["tasks"]["references"]["at"], graph["tasks"]["split"])}


@pytest.mark.parametrize("split,expected", FORMS)
def test_each_form_of_split_gives_the_members_written_out_in_two_builds_of_either_order(tmp_path, split, expected):
    (tmp_path / "back").mkdir()
    assert members(tmp_path, split) == expected
    assert members(tmp_path / "back", split, ROWS[::-1]) == expected


def test_the_forms_hash_group_and_graph_cut_the_one_hash_of_the_expression_hash():
    values = pa.table(dict(id=[row["id"] for row in ROWS], region=[row["region"] for row in ROWS],
                           world=[row["world"] for row in ROWS]))
    kinds = dict(id="identifier", region="identifier", world="identifier")
    for text, (_, expected) in (("hash(id, 'people')", FORMS[0]), ("hash(region, id, salt='people')", FORMS[1]),
                                ("hash(region, 'regions')", FORMS[2]), ("hash(world, 'worlds')", FORMS[3])):
        hashed = evaluate(text, values, kinds, plan=PLAN).values.to_pylist()
        assert {row["id"]: "train" if unit < 0.5 else "test" for row, unit in zip(ROWS, hashed)} == expected


def test_a_form_stops_on_shares_not_summing_to_one_an_unknown_name_a_missing_key_and_an_outcome(tmp_path):
    for split, message in (
            (dict(name="hash", key="id", parts=dict(train=0.5, test=0.4), salt="s"),
             "ask field split.parts states shares that sum to 1, found 0.9"),
            (dict(name="time", parts=dict(train=0.5, test=-0.5)), "ask field split.parts states the share of test"),
            (dict(name="random", parts=HALVES), "ask field split: no form of split is registered under the name"),
            (dict(name="hash", parts=HALVES, salt="s"), "ask field split holds fields it does not know or lacks: key"),
            (dict(name="hash", key="gap", parts=HALVES, salt="s"),
             "ask field split.key row 2 reads a missing value as a key"),
            (dict(name="hash", key="late", parts=HALVES, salt="s"),
             "ask field split.key reads the outcome column rows.late"),
            (dict(name="group", by="'a' if late else 'b'", parts=HALVES, salt="s"),
             "ask field split.by reads the outcome column rows.late"),
            (dict(name="time", by="day + (1 if late else 0)", parts=HALVES),
             "ask field split.by reads the outcome column rows.late"),
            (dict(name="time", parts=dict(train=None, test=None)), "only the first split states None"),
            (dict(name="time", parts=dict(train=None, test="'soon'")), "states a length of time above zero")):
        with pytest.raises(ValueError, match=message):
            members(tmp_path, split)
    with pytest.raises(ValueError, match="ask field split.key reads the outcome column rows.late"):
        found = declared(tmp_path, None)
        found["ask"] = tasks(found["flagged"], key="id", at="day", graph="world",
                             split=dict(name="hash", key="flag", parts=HALVES, salt="s"),
                             target=dict(kind="number", series="s", value="1", known="day + 1", about=None,
                                         utility=UTILITY),
                             references=dict(at=dict(sort="node", of=found["thing"], key="id", absent="stop")))
        mapped(tmp_path, found)


def regional(tasks, parameters):
    assert tasks.column_names == ["id", "day", "start", "region", "part", "world", "gap", "time", "graph"]
    return pa.array([parameters["first"] if region in ("n", "s") else "rest"
                     for region in tasks.column("region").to_pylist()])


def test_a_form_registered_by_a_script_reads_the_rows_without_their_outcomes_with_time_and_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(SPLITS, "held", {name: SPLITS.named(name) for name in SPLITS.names()})
    monkeypatch.setattr(SCRIPTED, "held", {})
    scripted("split", reason="no shipped form puts two regions apart", version=1)(regional)
    found = members(tmp_path, dict(name="regional", first="near"))
    assert found == dict(p0="near", p1="near", p2="near", p3="near", p4="rest", p5="rest", p6="rest", p7="rest")


def test_decisions_take_a_form_of_split_by_their_time(tmp_path):
    steps = table(tmp_path, "steps", [dict(student="s", step=step, shown=step) for step in (1, 2, 3)],
                  dict(student="identifier", step="number", shown=DAY))
    offers = table(tmp_path, "offers", [dict(student="s", step=step, type="h", shown=step) for step in (1, 2, 3)],
                   dict(student="identifier", step="number", type="category", shown=DAY))
    student = nodes(steps, distinct=["student"], holds="always", known=None)
    hinting = decisions(
        steps, key=["student", "step"], at="shown", actors="student", series=["correct"], about="student",
        utility=UTILITY, split=dict(name="time", parts=dict(train=None, score=1)),
        references=dict(student=dict(sort="node", of=student, key="student", absent="stop")),
        options=dict(rows=offers, decision=["student", "step"], key="type", chosen="True", available="True",
                     outcome=dict(correct="1.0"), known="shown + 1"),
        actions=dict(rows=offers, decision=["student", "step"], option="type", kind="'hint'", actor="student",
                     kinds=dict(hint=dict(roles={}))))
    found = mapped(tmp_path, dict(steps=steps, offers=offers, student=student, hinting=hinting))
    assert found["graphs"][0]["tasks"]["split"] == ["train", "train", "score"]
    batches = table(tmp_path, "batches", [dict(student="s", step=1, shown=1, batch=batch) for batch in ("a", "b")],
                    dict(student="identifier", step="number", shown=DAY, batch="identifier"))
    hashed = decisions(
        batches, key=["student", "step"], at="shown", actors="student", series=["correct"], about="student",
        utility=UTILITY, split=dict(name="hash", key="batch", parts=dict(train=0.5, score=0.5), salt="z"),
        references=dict(student=dict(sort="node", of=student, key="student", absent="stop")),
        options=dict(rows=offers, decision=["student", "step"], key="type", chosen="True", available="True",
                     outcome=dict(correct="1.0"), known="shown + 1"),
        actions=dict(rows=offers, decision=["student", "step"], option="type", kind="'hint'", actor="student",
                     kinds=dict(hint=dict(roles={}))))
    with pytest.raises(ValueError, match="decision s\\|1 field split: the rows of one decision"):
        mapped(tmp_path, dict(steps=steps, offers=offers, batches=batches, student=student, hinting=hashed))
