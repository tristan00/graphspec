from graphspec import edges, nodes, source, tasks

members = source(
    "members.csv",
    reader="csv",
    columns=dict(
        member="identifier",
        joined=dict(type="time", unit="day", format="%Y-%m-%d"),
        branch="category",
    ),
)
loans = source(
    "loans.csv",
    reader="csv",
    columns=dict(
        loan="identifier",
        member="identifier",
        book="identifier",
        lent=dict(type="time", unit="day", format="%Y-%m-%d"),
        back=dict(type="time", unit="day", format="%Y-%m-%d", outcome="back"),
        late=dict(type="flag", outcome="back"),
    ),
)
member = nodes(
    members,
    key="member",
    features=dict(branch="onehot"),
    holds=dict(name="state", since="joined", until=None),
    known=None,
)
book = nodes(loans, distinct=["book"], holds="always", known=None)
borrowed = edges(
    loans,
    source=dict(of=member, key="member", absent="stop"),
    target=dict(of=book, key="book", absent="stop"),
    directed=True,
    holds=dict(name="event", at="lent"),
    known=None,
)
overdue = tasks(
    loans,
    at="lent",
    references=dict(
        borrower=dict(sort="node", of=member, key="member", absent="stop"),
        title=dict(sort="node", of=book, key="book", absent="stop"),
    ),
    target=dict(
        kind="class",
        value="'late' if late else 'timely'",
        known="back",
        about=None,
        utility=dict(name="weighted", better="lower", weights=dict(late=1.0, timely=0.0)),
    ),
    split=dict(name="time", parts=dict(train=0.8, validation=0.2)),
)
