import logging
import time

from graphspec import decisions, nodes, scripted, source

DESCRIPTION = (
    "Decisions where only the chosen set of actions was logged, each action with targets in the graph "
    "and a number as parameter; the other options are drawn by a function"
)

stocks = source(
    "depots.csv",
    reader="csv",
    columns=dict(depot="identifier", day=dict(type="time", unit="day"), stock="number"),
)
fleet = source(
    "carriers.csv",
    reader="csv",
    columns=dict(carrier="identifier", capacity="number"),
)
shipments = source(
    "shipments.csv",
    reader="csv",
    columns=dict(
        day=dict(type="time", unit="day"),
        carrier="identifier",
        origin="identifier",
        destination="identifier",
        load="number",
        profit=dict(type="number", outcome="day + 7"),
    ),
)
depot = nodes(
    stocks,
    key="depot",
    features=dict(stock="standard"),
    holds=dict(name="change", at="day"),
    known=None,
)
carrier = nodes(fleet, key="carrier", features=dict(capacity="standard"), holds="always", known=None)


@scripted(
    "generator",
    reason=(
        "the description states that the other options are drawn by a function; no shipped generator "
        "replaces a target of a chosen action by another node its role may name"
    ),
    reads=("state", "chosen"),
    probable=False,
    version=1,
)
def rerouted(seen, parameters):
    started = time.perf_counter()
    pool = {}
    for option in seen.chosen:
        for place, action in enumerate(option["actions"]):
            for other in sorted(seen.state["nodes"]["depot"]):
                moved = dict(action, targets=dict(action["targets"], destination=[other]))
                actions = option["actions"][:place] + [moved] + option["actions"][place + 1 :]
                drawn = dict(actions=actions, together=option["together"], drawn=None)
                if seen.closed(drawn) is None and seen.fresh(drawn):
                    pool[seen.content(drawn)] = drawn
    tokens = sorted(pool)
    ranked = sorted(zip(seen.unit(tokens).tolist(), tokens))
    result = [pool[token] for draw, token in ranked[: parameters["count"]]]
    logging.getLogger(__name__).debug(
        "rerouted exit pool=%d drawn=%d seconds=%.4f", len(pool), len(result), time.perf_counter() - started
    )
    return result


dispatch = decisions(
    shipments,
    key=["day", "carrier"],
    at="day",
    references=dict(mover=dict(sort="node", of=carrier, key="carrier", absent="stop")),
    actors="mover",
    series=["profit"],
    about=None,
    utility=dict(name="weighted", better="higher", weights=dict(profit=1.0)),
    split=dict(name="time", parts=dict(train=0.6, validation=0.2, test=0.2)),
    generators=[dict(name="rerouted", seed=3, repeats=False, parameters=dict(count=4))],
    options=dict(
        rows=shipments,
        decision=["day", "carrier"],
        key="'logged'",
        chosen="True",
        available="True",
        together="'set'",
        outcome=dict(profit="sum(profit)"),
        known="day + 7",
    ),
    actions=dict(
        rows=shipments,
        decision=["day", "carrier"],
        option="'logged'",
        kind="'ship'",
        actor="carrier",
        kinds=dict(
            ship=dict(
                roles=dict(
                    origin=dict(sort="node", of=depot, keys="origin", count=[1, 1]),
                    destination=dict(sort="node", of=depot, keys="destination", count=[1, 1]),
                ),
                parameters=dict(load="standard"),
            ),
        ),
    ),
)
