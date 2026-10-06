from graphspec import edges, join, nodes, order, sets, source, tasks

DESCRIPTION = (
    "A league. `matches.parquet` has one row per match: its two teams, the kick-off, the final score. "
    "`lineups.parquet` has one row per player per match: the team, the position, and whether they started. "
    "`substitutions.parquet` has one row per substitution: the match, the minute, the player off and the "
    "player on. `ratings.parquet` has a rating per player per week, and a rating is sometimes revised in a "
    "later week: each row carries the week it is about and the week it was published. For each match, at "
    "kick-off, predict the difference of the final scores; a model must use the ratings as they were "
    "published by then, not as revised later. Matches are split by season, a column of `matches.parquet`."
)

matches = source(
    "matches.parquet",
    reader="parquet",
    columns=dict(
        match="identifier",
        home="identifier",
        away="identifier",
        kickoff=dict(type="time", unit="minute"),
        home_goals=dict(type="number", outcome="kickoff + span(3, 'hour')"),
        away_goals=dict(type="number", outcome="kickoff + span(3, 'hour')"),
        season="category",
    ),
)
lineups = source(
    "lineups.parquet",
    reader="parquet",
    columns=dict(
        match="identifier",
        player="identifier",
        team="identifier",
        position="category",
        started="flag",
    ),
)
substitutions = source(
    "substitutions.parquet",
    reader="parquet",
    columns=dict(
        match="identifier",
        minute=dict(type="duration", unit="minute"),
        off="identifier",
        on="identifier",
    ),
)
ratings = source(
    "ratings.parquet",
    reader="parquet",
    columns=dict(
        player="identifier",
        week=dict(type="time", unit="week", at="start"),
        published=dict(type="time", unit="week", at="end"),
        rating="number",
    ),
)
fielded = join(lineups, matches, on=dict(match="match"), take=dict(kickoff="kickoff"), unmatched="stop")
swapped = join(
    substitutions,
    matches,
    on=dict(match="match"),
    take=dict(kickoff="kickoff"),
    unmatched="stop",
)
versions = order(
    ratings,
    within=["player", "week"],
    by=["published"],
    gap=None,
    columns=dict(revised="following(published)"),
)
team = nodes(matches, distinct=["home", "away"], holds="always", known=None)
player = nodes(lineups, distinct=["player"], holds="always", known=None)
match = nodes(
    matches,
    key="match",
    links=dict(
        home=dict(of=team, key="home", absent="stop"),
        away=dict(of=team, key="away", absent="stop"),
    ),
    holds=dict(name="state", since="kickoff", until=None),
    known=None,
)
rating = nodes(
    versions,
    key=["player", "week"],
    features=dict(rating="standard"),
    links=dict(player=dict(of=player, key="player", absent="skip")),
    holds=dict(name="state", since="week", until=None),
    known="published",
    replaced="revised",
)
appearance = sets(
    fielded,
    key=["match", "player"],
    features=dict(position="onehot", started="value"),
    holds=dict(name="state", since="kickoff", until=None),
    known=None,
    members=[
        dict(
            rows=fielded,
            set=["match", "player"],
            node=dict(of=player, key="player", absent="stop"),
            role="'player'",
        ),
        dict(
            rows=fielded,
            set=["match", "player"],
            node=dict(of=team, key="team", absent="stop"),
            role="'team'",
        ),
        dict(
            rows=fielded,
            set=["match", "player"],
            node=dict(of=match, key="match", absent="stop"),
            role="'match'",
        ),
    ],
)
substitution = sets(
    swapped,
    key="row()",
    holds=dict(name="event", at="kickoff + minute"),
    known=None,
    members=[
        dict(rows=swapped, set="row()", node=dict(of=player, key="off", absent="stop"), role="'off'"),
        dict(rows=swapped, set="row()", node=dict(of=player, key="on", absent="stop"), role="'on'"),
        dict(rows=swapped, set="row()", node=dict(of=match, key="match", absent="stop"), role="'match'"),
    ],
)
result = edges(
    matches,
    source=dict(of=team, key="home", absent="stop"),
    target=dict(of=team, key="away", absent="stop"),
    directed=True,
    features=dict(home_goals="value", away_goals="value"),
    holds=dict(name="event", at="kickoff + span(3, 'hour')"),
    known=None,
)
margin = tasks(
    matches,
    at="kickoff",
    references=dict(
        match=dict(sort="node", of=match, key="match", absent="stop"),
        home=dict(sort="node", of=team, key="home", absent="stop"),
        away=dict(sort="node", of=team, key="away", absent="stop"),
    ),
    target=dict(
        kind="number",
        series="margin",
        value="home_goals - away_goals",
        known="kickoff + span(3, 'hour')",
        about="match",
        utility=dict(name="weighted", better="higher", weights=1.0),
    ),
    split="season",
)
