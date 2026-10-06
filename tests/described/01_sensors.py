from graphspec import compute, edges, nodes, select, source, tasks, unpivot

DESCRIPTION = (
    "Sensors joined by a kernel of road distance above a threshold; a reading every five minutes; "
    "predict the next twelve"
)

distances = source(
    "distances.csv",
    reader="csv",
    columns=dict(origin="identifier", destination="identifier", metres="number"),
)
wide = source(
    "readings.h5",
    reader="hdf5",
    dataset="speed",
    axis="minute",
    columns={
        "minute": dict(type="time", unit="minute"),
        "*": dict(type="number", nulls=[0.0], missing="omit"),
    },
    through="max(minute)",
)
readings = unpivot(wide, keep=["minute"], name="sensor", value="speed")
kernel = compute(distances, columns=dict(weight="exp(-((metres / deviation(metres)) ** 2))"))
close = select(kernel, where="weight >= 0.1")
sensor = nodes(
    readings,
    key="sensor",
    features=dict(speed="standard"),
    holds=dict(name="change", at="minute"),
    known=None,
)
road = edges(
    close,
    source=dict(of=sensor, key="origin", absent="skip"),
    target=dict(of=sensor, key="destination", absent="skip"),
    directed=True,
    features=dict(weight="value"),
    holds="always",
    known=None,
)
ahead = tasks(
    readings,
    at="minute",
    references=dict(place=dict(sort="node", of=sensor, key="sensor", absent="stop")),
    target=dict(
        kind="path",
        series=["speed"],
        value="points(later.minute - minute, 'speed', later.speed)",
        later=dict(rows=readings, on=dict(sensor="sensor"), at="minute", within=60),
        known="minute + 60",
        about="place",
        utility=dict(name="weighted", better="higher", weights=1.0),
    ),
    split=dict(name="time", parts=dict(train=0.7, validation=0.1, test=0.2)),
)
