import math

from graphspec.infra.timing import timed

from .measures import MEASURES

SERIES = "series/%s/loss"
MEASURE = "measure/%s"


@timed("objective.weights")
def weights(names, given):
    if not given:
        return None
    parsed = dict.fromkeys(names, 0.0)
    named = set()
    for item in given:
        name, _, share = item.partition("=")
        if name in named or name not in parsed:
            raise ValueError("objective names each series of %s at most once" % ", ".join(names))
        named.add(name)
        parsed[name] = float(share)
    if (
        any(not math.isfinite(share) or share < 0 for share in parsed.values())
        or not sum(parsed.values()) > 0
    ):
        raise ValueError("objective weights must be finite, nonnegative and not all zero")
    return parsed


@timed("objective.ranking")
def ranking(names):
    entry = MEASURES.named(names[0])
    if entry.better != "lower":
        raise ValueError(
            "a search ranks its trials by the first measure of its statement, which states that lower is better; "
            "%s states that %s is better" % (names[0], entry.better)
        )
    return entry


@timed("objective.blended")
def blended(series, chosen):
    return sum(share * series[name] for name, share in chosen.items()) / sum(chosen.values())


@timed("objective.earlier")
def earlier(result, plan):
    if plan["objective"] is None:
        return result["raw_val_mse"]
    return blended({name: entry["val_loss"] for name, entry in result["series"].items()}, plan["objective"])


@timed("objective.value")
def value(result, plan):
    if "measures" not in plan:
        return earlier(result, plan)
    held = result["measures"][plan["measures"][0]]
    if plan["objective"] is None:
        return held["value"]
    return blended(held["series"], plan["objective"])


@timed("objective.measures")
def measures(plan):
    if "measures" not in plan:
        if plan["objective"] is None:
            return [
                dict(key="mse", label="Validation MSE", short="MSE", better="lower", shown=True),
                dict(key="r2", label="Validation R²", short="R²", better="higher", shown=True),
            ]
        first = [dict(key="mse", label="Validation loss", short="Loss", better="lower", shown=True)]
    else:
        ranked = ranking(plan["measures"])
        first = [dict(key="mse", label=ranked.label, short=ranked.short, better="lower", shown=True)]
        first += [
            dict(
                key=MEASURE % name,
                label=MEASURES.named(name).label,
                short=MEASURES.named(name).short,
                better=MEASURES.named(name).better,
                shown=True,
            )
            for name in plan["measures"][1:]
        ]
    if plan["objective"] is None:
        return first
    return first + [
        dict(
            key=SERIES % name,
            label=name + " " + first[0]["label"].lower(),
            short=name + " " + first[0]["short"].lower(),
            better="lower",
            shown=False,
        )
        for name in plan["objective"]
    ]


@timed("objective.columns")
def columns(plan):
    return [] if plan["objective"] is None else [SERIES % name for name in plan["objective"]]


@timed("objective.scores")
def scores(result, plan):
    if "measures" not in plan:
        if plan["objective"] is None or "series" not in result:
            return {}
        return {SERIES % name: result["series"][name]["val_loss"] for name in plan["objective"]}
    if "measures" not in result:
        return {}
    found = {MEASURE % name: result["measures"][name]["value"] for name in plan["measures"][1:]}
    if plan["objective"] is not None:
        held = result["measures"][plan["measures"][0]]["series"]
        found.update({SERIES % name: held[name] for name in plan["objective"]})
    return found
