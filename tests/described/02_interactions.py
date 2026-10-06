from graphspec import edges, nodes, source, tasks

DESCRIPTION = "Interactions of users with items over time; predict the next item among sampled negatives"

visits = source(
    "interactions.csv",
    reader="csv",
    columns=dict(user="identifier", item="identifier", second=dict(type="time", unit="second")),
)
user = nodes(visits, distinct=["user"], holds="always", known=None)
item = nodes(visits, distinct=["item"], holds="always", known=None)
chose = edges(
    visits,
    source=dict(of=user, key="user", absent="stop"),
    target=dict(of=item, key="item", absent="stop"),
    directed=True,
    holds=dict(name="event", at="second"),
    known=None,
)
following = tasks(
    visits,
    at="second",
    references=dict(who=dict(sort="node", of=user, key="user", absent="stop")),
    target=dict(
        kind="link",
        series="next",
        relation=chose,
        value="first(later.item, by=later.second)",
        later=dict(rows=visits, on=dict(user="user"), at="second", within=None),
        known="min(later.second)",
        about="who",
        utility=dict(name="weighted", better="higher", weights=1.0),
    ),
    split=dict(name="time", parts=dict(train=0.7, validation=0.15, test=0.15)),
)
