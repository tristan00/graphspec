from graphspec import decisions, expand, join, nodes, source

DESCRIPTION = (
    "A dispatch centre. `calls.jsonl` has one record per call: its time, its place, its priority, the "
    "dispatcher who took it, and a list of the vehicles that were free at that moment, each with its "
    "station; the vehicle the dispatcher sent is marked in that list. `vehicles.csv` gives each vehicle its "
    "type and station; `stations.csv` each station's place. `arrivals.csv` gives, per call, the minutes "
    "until the vehicle arrived, known only when it arrives. Several calls can come at the same moment and "
    "are then taken by different dispatchers, none of whom knows what the others choose. The model is to "
    "learn, for each call, which of the free vehicles to send: every free vehicle is an option, the one "
    "sent is the one chosen, and the minutes until arrival are the outcome of the chosen option alone."
)

calls = source(
    "calls.jsonl",
    reader="json",
    columns=dict(
        call="identifier",
        time=dict(type="time", unit="second", format="%Y-%m-%dT%H:%M:%S"),
        place=dict(type="point", width=2),
        priority="number",
        dispatcher="identifier",
        free=dict(type="list", of="nested"),
    ),
)
vehicles = source(
    "vehicles.csv",
    reader="csv",
    columns=dict(vehicle="identifier", type="category", station="identifier"),
)
stations = source(
    "stations.csv",
    reader="csv",
    columns=dict(station="identifier", place=dict(type="point", width=2, separator=" ")),
)
arrivals = source(
    "arrivals.csv",
    reader="csv",
    columns=dict(call="identifier", minutes=dict(type="duration", unit="minute")),
)
offers = expand(
    calls,
    column="free",
    columns=dict(vehicle="identifier", station="identifier", sent="flag"),
    names=dict(place="rank"),
)
answered = join(
    offers,
    arrivals,
    on=dict(call="call"),
    take=dict(minutes="minutes"),
    unmatched="keep",
    outcomes=dict(minutes="time + minutes"),
)
dispatcher = nodes(calls, distinct=["dispatcher"], holds="always", known=None)
station = nodes(stations, key="station", features=dict(place="standard"), holds="always", known=None)
vehicle = nodes(
    vehicles,
    key="vehicle",
    features=dict(type="onehot"),
    links=dict(station=dict(of=station, key="station", absent="stop")),
    holds="always",
    known=None,
)
dispatch = decisions(
    calls,
    key="call",
    at="time",
    references=dict(taker=dict(sort="node", of=dispatcher, key="dispatcher", absent="stop")),
    features=dict(place="standard", priority="value"),
    actors="taker",
    series=["minutes"],
    about=None,
    utility=dict(name="weighted", better="lower", weights=1.0),
    split="'train'",
    concurrent="True",
    options=dict(
        rows=answered,
        decision="call",
        key="vehicle",
        chosen="sent",
        available="True",
        outcome=dict(minutes="minutes if sent else missing"),
        known="time + minutes",
    ),
    actions=dict(
        rows=answered,
        decision="call",
        option="vehicle",
        kind="'send'",
        actor="dispatcher",
        kinds=dict(
            send=dict(
                roles=dict(
                    vehicle=dict(sort="node", of=vehicle, keys="vehicle", count=[1, 1]),
                    origin=dict(sort="node", of=station, keys="station", count=[1, 1]),
                ),
            ),
        ),
    ),
)
