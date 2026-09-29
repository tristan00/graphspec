import math

from . import catalog
from .catalog import TRACKED

SERIES = "series/%s/loss"
SPREAD = "series/%s/spread"
FLOOR = 1e-6
MEASURE = "series/%s/%s"


def weights(bank, given):
    if bank.level != "path":
        if given:
            raise ValueError("only path targets have series to weight")
        return None
    if not given:
        return {name: 1.0 for name in bank.series_names}
    parsed = dict.fromkeys(bank.series_names, 0.0)
    named = set()
    for item in given:
        name, _, share = item.partition("=")
        if name in named or name not in parsed:
            raise ValueError("objective names each series of %s at most once" % ", ".join(bank.series_names))
        named.add(name)
        parsed[name] = float(share)
    if (
        any(not math.isfinite(share) or share < 0 for share in parsed.values())
        or not sum(parsed.values()) > 0
    ):
        raise ValueError("objective weights must be finite, nonnegative and not all zero")
    return parsed


def value(result, plan):
    chosen = plan["objective"]
    if chosen is None:
        return result["raw_val_mse"]
    found = blend(result, chosen, "val_loss")
    if plan.get("spread") is None:
        return found
    return found - plan["spread"] * math.log2(max(blend(result, chosen, "spread"), FLOOR))


def blend(result, chosen, key):
    series = result["series"]
    return sum(share * series[name][key] for name, share in chosen.items()) / sum(chosen.values())


def ranked(plan):
    if plan.get("spread") is None:
        return [dict(key="mse", label="Validation loss", short="Loss", better="lower", shown=True)]
    return [
        dict(
            key="mse",
            label="Score: validation loss − %s · log2 spread" % plan["spread"],
            short="Score",
            better="lower",
            shown=True,
        ),
        dict(key="loss", label="Validation loss", short="Loss", better="lower", shown=True),
        dict(
            key="spread",
            label="Spread of predictions over the blocks drawn at one pause",
            short="Spread",
            better="higher",
            shown=True,
        ),
    ] + [
        dict(key=SPREAD % name, label=name + " spread", short=name + " spread", better="higher", shown=False)
        for name in plan["objective"]
    ]


def measures(plan):
    if plan["objective"] is None:
        return [
            dict(key="mse", label="Validation MSE", short="MSE", better="lower", shown=True),
            dict(key="r2", label="Validation R²", short="R²", better="higher", shown=True),
        ]
    return (
        ranked(plan)
        + [
            dict(
                key=SERIES % name,
                label=name + " validation loss",
                short=name + " loss",
                better="lower",
                shown=False,
            )
            for name in plan["objective"]
        ]
        + [
            dict(
                key=MEASURE % (name, key),
                label="%s %s" % (name, catalog.entry(key)["label"]),
                short="%s %s" % (name, key.split("/")[1]),
                better=catalog.entry(key)["better"],
                shown=False,
            )
            for name in plan["objective"]
            for key in TRACKED
        ]
    )


def columns(plan):
    return [] if plan["objective"] is None else [SERIES % name for name in plan["objective"]]


def scores(result, plan):
    if plan["objective"] is None or "series" not in result:
        return {}
    found = {SERIES % name: result["series"][name]["val_loss"] for name in plan["objective"]}
    for name in plan["objective"]:
        for key, value in result["series"][name]["measures"].items():
            found[MEASURE % (name, key)] = value
    if plan.get("spread") is not None:
        found.update(
            loss=blend(result, plan["objective"], "val_loss"),
            spread=blend(result, plan["objective"], "spread"),
            **{SPREAD % name: result["series"][name]["spread"] for name in plan["objective"]},
        )
    return found
