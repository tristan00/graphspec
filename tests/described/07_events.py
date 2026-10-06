from graphspec import join, nodes, sets, source, tasks

DESCRIPTION = "Events that each touch several objects of several kinds, each with a role"

things = source(
    "objects.csv",
    reader="csv",
    columns=dict(id="identifier", sort="category", weight=dict(type="number", missing="omit")),
)
happenings = source(
    "events.csv",
    reader="csv",
    columns=dict(id="identifier", activity="category", minute=dict(type="time", unit="minute")),
)
touches = source(
    "touches.csv",
    reader="csv",
    columns=dict(event="identifier", object="identifier", role="category"),
)
thing = nodes(
    things,
    key="id",
    kind="sort",
    features=dict(weight="standard"),
    holds="always",
    known=None,
)
event = sets(
    happenings,
    key="id",
    features=dict(activity="onehot"),
    holds=dict(name="event", at="minute"),
    known=None,
    members=[
        dict(rows=touches, set="event", node=dict(of=thing, key="object", absent="stop"), role="role"),
    ],
)
timed = join(
    touches,
    happenings,
    on=dict(id="event"),
    take=dict(minute="minute", activity="activity"),
    unmatched="stop",
)
coming = tasks(
    timed,
    at="minute",
    references=dict(subject=dict(sort="node", of=thing, key="object", absent="stop")),
    target=dict(
        kind="class",
        value="first(later.activity, by=later.minute)",
        later=dict(rows=timed, on=dict(object="object"), at="minute", within=None),
        known="min(later.minute)",
        about="subject",
        utility=dict(name="weighted", better="higher", weights=1.0),
    ),
    split=dict(name="hash", key="object", parts=dict(train=0.8, validation=0.2), salt="events"),
)
