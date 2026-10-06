from graphspec import decisions, edges, nodes, source

DESCRIPTION = "A game where several players act at once and each sees part of the state"

sightings = source(
    "sightings.parquet",
    reader="parquet",
    columns=dict(
        game="identifier",
        turn=dict(type="time", unit="turn"),
        unit="identifier",
        owner="identifier",
        strength="number",
        watchers=dict(type="list", of="identifier"),
    ),
)
orders = source(
    "orders.parquet",
    reader="parquet",
    columns=dict(
        game="identifier",
        turn=dict(type="time", unit="turn"),
        player="identifier",
        unit="identifier",
        move="category",
        onto=dict(type="identifier", missing="omit"),
        force=dict(type="number", missing="omit"),
        gain=dict(type="number", outcome="turn + 1"),
    ),
)
player = nodes(sightings, distinct=["owner"], graph="game", holds="always", known=None)
unit = nodes(
    sightings,
    key="unit",
    graph="game",
    features=dict(strength="standard"),
    holds=dict(name="snapshot", at="turn", between="absent"),
    known=None,
    seen="watchers",
)
owned = edges(
    sightings,
    graph="game",
    source=dict(of=unit, key="unit", absent="stop"),
    target=dict(of=player, key="owner", absent="stop"),
    directed=True,
    holds=dict(name="snapshot", at="turn", between="absent"),
    known=None,
    seen="watchers",
)
turn = decisions(
    orders,
    key=["turn", "player"],
    graph="game",
    at="turn",
    observer="player",
    references=dict(side=dict(sort="node", of=player, key="player", absent="stop")),
    actors="side",
    concurrent="turn",
    series=["gain"],
    about=None,
    utility=dict(name="weighted", better="higher", weights=dict(gain=1.0)),
    split=dict(name="graph", parts=dict(train=0.6, validation=0.2, test=0.2), salt="game"),
    generators=[dict(name="others", seed=7, repeats=False, parameters=dict(count=3))],
    options=dict(
        rows=orders,
        decision=["turn", "player"],
        key="'played'",
        chosen="True",
        available="True",
        together="'set'",
        known="turn + 1",
    ),
    actions=dict(
        rows=orders,
        decision=["turn", "player"],
        option="'played'",
        kind="move",
        actor="player",
        outcome=dict(gain="gain"),
        kinds=dict(
            strike=dict(
                roles=dict(
                    by=dict(sort="node", of=unit, keys="unit", count=[1, 1]),
                    onto=dict(sort="node", of=unit, keys="onto", count=[1, 1]),
                ),
                parameters=dict(force="standard"),
            ),
            hold=dict(
                roles=dict(by=dict(sort="node", of=unit, keys="unit", count=[1, 1])),
                parameters=dict(),
            ),
        ),
    ),
)
