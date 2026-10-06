from graphspec import expand, nodes, sets, source, tasks

DESCRIPTION = "Papers joining all their authors as one edge; predict the next set"

works = source(
    "papers.json",
    reader="json",
    columns=dict(
        id="identifier",
        lab="identifier",
        year=dict(type="time", unit="year"),
        authors=dict(type="list", of="identifier"),
    ),
)
bylines = expand(works, column="authors", into="author")
author = nodes(bylines, distinct=["author"], holds="always", known=None)
paper = sets(
    works,
    key="id",
    holds=dict(name="event", at="year"),
    known=None,
    members=[
        dict(rows=bylines, set="id", node=dict(of=author, key="author", absent="stop"), role="'author'"),
    ],
)
team = tasks(
    works,
    at="year",
    references=dict(last=dict(sort="node", of=paper, key="id", absent="stop")),
    target=dict(
        kind="set",
        series="next",
        sets=paper,
        value=dict(author="first(later.authors, by=later.year)"),
        later=dict(rows=works, on=dict(lab="lab"), at="year", within=None),
        known="min(later.year)",
        about=None,
        utility=dict(name="weighted", better="higher", weights=1.0),
    ),
    split=dict(name="time", parts=dict(train=0.7, validation=0.15, test=0.15)),
)
