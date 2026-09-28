import math

SERIES = "series/%s/loss"


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
    series = result["series"]
    return sum(share * series[name]["val_loss"] for name, share in chosen.items()) / sum(chosen.values())


def measures(plan):
    if plan["objective"] is None:
        return [
            dict(key="mse", label="Validation MSE", short="MSE", better="lower", shown=True),
            dict(key="r2", label="Validation R²", short="R²", better="higher", shown=True),
        ]
    return [dict(key="mse", label="Validation loss", short="Loss", better="lower", shown=True)] + [
        dict(
            key=SERIES % name,
            label=name + " validation loss",
            short=name + " loss",
            better="lower",
            shown=False,
        )
        for name in plan["objective"]
    ]


def columns(plan):
    return [] if plan["objective"] is None else [SERIES % name for name in plan["objective"]]


def scores(result, plan):
    if plan["objective"] is None or "series" not in result:
        return {}
    return {SERIES % name: result["series"][name]["val_loss"] for name in plan["objective"]}
