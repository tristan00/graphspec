from graphspec import edges, join, nodes, source, tasks

DESCRIPTION = "Papers citing papers; a paper sees only older papers; predict the venue"

works = source(
    "papers.parquet",
    reader="parquet",
    columns=dict(
        id="identifier",
        year=dict(type="time", unit="year"),
        words=dict(type="vector", width=16),
        venue=dict(type="category", outcome=None),
    ),
)
listed = source(
    "citations.parquet",
    reader="parquet",
    columns=dict(citing="identifier", cited="identifier"),
)
dated = join(listed, works, on=dict(id="citing"), take=dict(year="year"), unmatched="stop")
paper = nodes(
    works,
    key="id",
    features=dict(words="value"),
    holds=dict(name="state", since="year", until=None),
    known=None,
)
cites = edges(
    dated,
    source=dict(of=paper, key="citing", absent="stop"),
    target=dict(of=paper, key="cited", absent="skip"),
    directed=True,
    holds=dict(name="state", since="year", until=None),
    known=None,
)
venue = tasks(
    works,
    at="year",
    references=dict(work=dict(sort="node", of=paper, key="id", absent="stop")),
    target=dict(
        kind="class",
        value="venue",
        known=None,
        about="work",
        utility=dict(name="weighted", better="higher", weights=1.0),
    ),
    split="'train' if year < 2018 else 'validation' if year < 2019 else 'test'",
)
