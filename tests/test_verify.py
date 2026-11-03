import json

import pytest

from graphspec import Builder, BuildPlan, edges, nodes, source, tasks, verify
from tuning import build_bank

pytestmark = [pytest.mark.xdist_group("verify"), pytest.mark.builder]
PLAN = BuildPlan(16 << 20, 1)
DAY = dict(type="time", unit="day")
UTILITY = dict(name="weighted", better="higher", weights=1.0)
PEOPLE = [dict(id="a", day=1, score=1.0), dict(id="b", day=2, score=2.0), dict(id="c", day=3, score=3.0)]
LINKS = [dict(left="a", right="b", day=3)]
PLACES = dict(person=("people.json",), knew=("links.json",), ask=("people.json",))


def table(folder, name, rows, columns):
    (folder / (name + ".json")).write_text(json.dumps(rows), encoding="utf-8")
    return source(name + ".json", "json", columns)


def script(folder, people, links):
    people = table(folder, "people", people, dict(id="identifier", day=DAY, score="number"))
    links = table(folder, "links", links, dict(left="identifier", right="identifier", day=DAY))
    person = nodes(people, key="id", holds=dict(name="state", since="day", until=None), known=None,
                   features=dict(score="value"))
    knew = edges(links, source=dict(of=person, key="left", absent="stop"),
                 target=dict(of=person, key="right", absent="stop"), directed=True,
                 holds=dict(name="event", at="day"), known=None)
    ask = tasks(people, at="day", references=dict(who=dict(sort="node", of=person, key="id", absent="stop")),
                target=dict(kind="number", series="score", value="score", known="day + 1", about=None,
                            utility=UTILITY), split="'train'")
    return Builder(dict(people=people, links=links, person=person, knew=knew, ask=ask), folder)


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    folder = tmp_path_factory.mktemp("verified")
    build_bank(script(folder, PEOPLE, LINKS), folder / "bank", 1, ("train",), plan=PLAN)
    return (folder / "bank").resolve()


def test_an_unchanged_source_verifies_clean_and_one_seed_samples_the_same_times(bank, tmp_path):
    builder = script(tmp_path, PEOPLE, LINKS)
    first, again, other = verify(builder, bank, 2, 7), verify(builder, bank, 2, 7), verify(builder, bank, 2, 1)
    assert first == again and first.differences == () and other.differences == ()
    assert (first.times, other.times) == ((1, 3), (2, 3))
    assert str(first) == "the build of the bank %s holds what its sources hold, at the 2 sampled times 1, 3" % bank
    reordered = verify(script(tmp_path, PEOPLE[::-1], LINKS), bank, 9, 7)
    assert reordered.times == (1, 2, 3) and reordered.differences == ()


def test_a_source_changed_after_the_build_is_reported_with_the_declaration_the_source_the_row_and_the_times(
        bank, tmp_path):
    changed = [dict(PEOPLE[0], score=9.0), PEOPLE[1], dict(id="d", day=5, score=4.0)]
    result = verify(script(tmp_path, changed, [dict(LINKS[0], day=4)]), bank, 9, 7)
    assert result.times == (1, 2, 3, 4, 5)
    assert [(found.declaration, found.places, found.rows, found.times, found.message)
            for found in result.differences] == [
        ("person", PLACES["person"], (), (3, 4, 5),
         "the build holds the node c of the kind person from the time 3, which no row of the source states"),
        ("person", PLACES["person"], (0,), (1, 2, 3, 4, 5),
         "the node a of the kind person from the time 1: the column score is 9.0 in the source and 1.0 in the build"),
        ("person", PLACES["person"], (2,), (5,),
         "the source states the node d of the kind person from the time 5, which the build does not hold"),
        ("knew", PLACES["knew"], (), (3,),
         "the build holds the edge of knew from a to b from the time 3, which no row of the source states"),
        ("knew", PLACES["knew"], (0,), (4,),
         "the source states the edge of knew from a to b from the time 4, which the build does not hold"),
        ("ask", PLACES["ask"], (), (3,),
         "the build holds the task of the kind ask naming {\"who\":\"c\"} at the time 3, which no row of the "
         "source states"),
        ("ask", PLACES["ask"], (0,), (1,),
         "the task of the kind ask naming {\"who\":\"a\"} at the time 1: target is 9.0 in the source and 1.0 in "
         "the build"),
        ("ask", PLACES["ask"], (2,), (5,),
         "the source states the task of the kind ask naming {\"who\":\"d\"} at the time 5, which the build does "
         "not hold")]
    text = str(result).splitlines()
    assert text[0] == "the sources differ from the build of the bank %s in 8 facts, at the 5 sampled times " \
                      "1, 2, 3, 4, 5" % bank
    assert text[1:5] == ["declaration person, reading people.json",
                         "  seen at 3, 4, 5: " + result.differences[0].message,
                         "  row 0, seen at 1, 2, 3, 4, 5: " + result.differences[1].message,
                         "  row 2, seen at 5: " + result.differences[2].message]


def test_times_and_seed_are_stated_and_a_graph_stated_by_calls_has_no_source_to_read_again(bank, tmp_path):
    builder = script(tmp_path, PEOPLE, LINKS)
    for times, seed, field in ((0, 7, "times"), (2.0, 7, "times"), (2, None, "seed"), (2, "7", "seed")):
        with pytest.raises(ValueError, match="verify states %s" % field):
            verify(builder, bank, times, seed)
    with pytest.raises(ValueError, match="the builder holds no mapping"):
        verify(Builder({}, tmp_path), bank, 2, 7)
