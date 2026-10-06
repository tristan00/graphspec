from graphspec import edges, nodes, source, tasks

DESCRIPTION = "Trips as ordered sequences of road segments; predict the duration of a trip"

roads = source(
    "segments.csv",
    reader="csv",
    columns=dict(id="identifier", origin="identifier", destination="identifier", metres="number"),
)
rides = source(
    "trips.json",
    reader="json",
    columns=dict(
        trip="identifier",
        left=dict(type="time", unit="minute"),
        route=dict(type="list", of="identifier"),
        took=dict(type="duration", unit="minute", outcome="left + took"),
    ),
)
junction = nodes(roads, distinct=["origin", "destination"], holds="always", known=None)
segment = edges(
    roads,
    key="id",
    source=dict(of=junction, key="origin", absent="stop"),
    target=dict(of=junction, key="destination", absent="stop"),
    directed=True,
    features=dict(metres="standard"),
    holds="always",
    known=None,
)
lasted = tasks(
    rides,
    at="left",
    references=dict(way=dict(sort="path", of=segment, keys="route", absent="stop")),
    target=dict(
        kind="number",
        series="minutes",
        value="took",
        known="left + took",
        about=None,
        utility=dict(name="weighted", better="lower", weights=dict(minutes=1.0)),
    ),
    split=dict(name="time", parts=dict(train=0.7, validation=0.1, test=0.2)),
)
