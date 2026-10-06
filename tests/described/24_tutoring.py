from graphspec import decisions, edges, expand, nodes, order, source

DESCRIPTION = (
    "A tutoring system. `sessions.jsonl` has one record per session of a student: the student, the start, "
    "and a list of steps; each step holds the exercise shown, the hints the tutor could have given at that "
    "step (a list, each with its type and its length in words), the one hint it gave, and whether the "
    "student then answered correctly, which is known when the step ends, twenty to ninety seconds later, "
    "the time of the answer being in the step. `exercises.csv` gives each exercise its topic and "
    "difficulty, and `prerequisites.csv` lists pairs of topics, one needed before the other. Giving no hint "
    "is always possible and is logged as such. The model is to learn, at each step, which hint to give: the "
    "hints the tutor could have given and giving none are the options, the one given is the one chosen, and "
    "whether the answer was correct is the outcome of the chosen option alone. The sessions of the last "
    "four weeks of the data are for scoring."
)

sessions = source(
    "sessions.jsonl",
    reader="json",
    columns=dict(
        student="identifier",
        start=dict(type="time", unit="second", format="%Y-%m-%dT%H:%M:%S"),
        steps=dict(type="list", of="nested"),
    ),
)
exercises = source(
    "exercises.csv",
    reader="csv",
    columns=dict(exercise="identifier", topic="identifier", difficulty="number"),
)
prerequisites = source(
    "prerequisites.csv",
    reader="csv",
    columns=dict(topic="identifier", needed="identifier"),
)
steps = expand(
    sessions,
    column="steps",
    columns=dict(
        exercise="identifier",
        hints=dict(type="list", of="nested"),
        given=dict(type="category", outcome="answered"),
        correct=dict(type="flag", outcome="answered"),
        answered=dict(type="time", unit="second", format="%Y-%m-%dT%H:%M:%S", outcome="answered"),
    ),
    names=dict(place="step"),
)
timed = order(
    steps,
    within=["student", "start"],
    by=["step"],
    gap=None,
    columns=dict(shown="coalesce(previous(answered), start)"),
)
offers = expand(steps, column="hints", columns=dict(type="category", words="number"))
student = nodes(sessions, distinct=["student"], holds="always", known=None)
tutor = nodes(sessions, distinct=["'tutor'"], holds="always", known=None)
exercise = nodes(
    exercises,
    key="exercise",
    features=dict(difficulty="standard"),
    links=dict(topic="topic"),
    holds="always",
    known=None,
)
requires = edges(
    prerequisites,
    source=dict(of=exercise.topic, key="topic", absent="create"),
    target=dict(of=exercise.topic, key="needed", absent="create"),
    directed=True,
    holds="always",
    known=None,
)
attempt = edges(
    steps,
    source=dict(of=student, key="student", absent="stop"),
    target=dict(of=exercise, key="exercise", absent="stop"),
    directed=True,
    features=dict(given="onehot", correct="value"),
    holds=dict(name="event", at="answered"),
    known=None,
)
hinting = decisions(
    timed,
    key=["student", "start", "step"],
    at="shown",
    references=dict(
        student=dict(sort="node", of=student, key="student", absent="stop"),
        exercise=dict(sort="node", of=exercise, key="exercise", absent="stop"),
        tutor=dict(sort="node", of=tutor, key="'tutor'", absent="stop"),
    ),
    actors="tutor",
    series=["correct"],
    about="student",
    utility=dict(name="weighted", better="higher", weights=dict(correct=1.0)),
    split=dict(name="time", by="start", parts=dict(train=None, score="span(4, 'week')")),
    nothing=dict(
        key="'none'",
        chosen="given == 'none'",
        available="True",
        outcome=dict(correct="number(correct) if given == 'none' else missing"),
        known="answered",
    ),
    options=dict(
        rows=offers,
        decision=["student", "start", "step"],
        key="type",
        chosen="type == given",
        available="True",
        outcome=dict(correct="number(correct) if type == given else missing"),
        known="answered",
    ),
    actions=dict(
        rows=offers,
        decision=["student", "start", "step"],
        option="type",
        kind="'hint'",
        actor="'tutor'",
        kinds=dict(hint=dict(parameters=dict(type="onehot", words="standard"))),
    ),
)
