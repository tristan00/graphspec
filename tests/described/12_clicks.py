from graphspec import edges, nodes, order, source, tasks

DESCRIPTION = "Clicks cut into sessions by a gap of thirty minutes, with edges between consecutive clicks"

clicks = source(
    "clicks.csv",
    reader="csv",
    columns=dict(
        id="identifier",
        user="identifier",
        page="category",
        minute=dict(type="time", unit="minute", format="%Y-%m-%d %H:%M"),
    ),
    through="max(minute)",
)
sessions = order(
    clicks,
    within=["user"],
    by=["minute"],
    gap=30,
    columns=dict(before="previous(id)"),
)
click = nodes(
    sessions,
    key="id",
    features=dict(page="onehot", place="value"),
    holds=dict(name="state", since="minute", until=None),
    known=None,
)
follows = edges(
    sessions,
    where="before is not missing",
    source=dict(of=click, key="before", absent="stop"),
    target=dict(of=click, key="id", absent="stop"),
    directed=True,
    holds=dict(name="state", since="minute", until=None),
    known=None,
)
stays = tasks(
    sessions,
    at="minute",
    references=dict(last=dict(sort="node", of=click, key="id", absent="stop")),
    target=dict(
        kind="class",
        value="'stays' if count(later.id) > 0 else 'leaves'",
        later=dict(rows=sessions, on=dict(user="user"), at="minute", within=30),
        known="minute + 30",
        about="last",
        utility=dict(name="weighted", better="higher", weights=dict(stays=1.0, leaves=0.0)),
    ),
    split=dict(name="hash", key="user", parts=dict(train=0.8, validation=0.2), salt="clicks"),
)
