from graphspec import edges, nodes, source, tasks

DESCRIPTION = (
    "Accounts and transfers as parallel edges with amounts; the label of a transfer arrives thirty days "
    "later and is then known to later tasks"
)

ledger = source(
    "transfers.csv",
    reader="csv",
    columns=dict(
        id="identifier",
        sender="identifier",
        receiver="identifier",
        day=dict(type="time", unit="day", format="%Y-%m-%d"),
        amount="number",
        laundering=dict(type="flag", outcome="day + 30"),
    ),
)
account = nodes(ledger, distinct=["sender", "receiver"], holds="always", known=None)
transfer = edges(
    ledger,
    source=dict(of=account, key="sender", absent="stop"),
    target=dict(of=account, key="receiver", absent="stop"),
    directed=True,
    features=dict(amount="logarithm"),
    holds=dict(name="event", at="day"),
    known=None,
    ties=dict(name="sequence", seq="row()"),
)
labelled = edges(
    ledger,
    source=dict(of=account, key="sender", absent="stop"),
    target=dict(of=account, key="receiver", absent="stop"),
    directed=True,
    features=dict(laundering="value"),
    holds=dict(name="state", since="day + 30", until=None),
    known=None,
    ties=dict(name="sequence", seq="row()"),
)
suspect = tasks(
    ledger,
    at="day",
    references=dict(
        moved=dict(sort="edge", of=transfer, source="sender", target="receiver", absent="stop"),
    ),
    features=dict(amount="logarithm"),
    target=dict(
        kind="class",
        value="'laundering' if laundering else 'clean'",
        known="day + 30",
        about="moved",
        utility=dict(name="weighted", better="higher", weights=dict(laundering=1.0, clean=0.0)),
    ),
    split=dict(name="time", parts=dict(train=0.6, validation=0.2, test=0.2)),
)
