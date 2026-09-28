import math


def weights(bank, given):
    if bank.level != "path":
        if given:
            raise ValueError("only path targets have series to weight")
        return None
    if not given:
        return {name: 1.0 for name in bank.series_names}
    parsed = {}
    for item in given:
        name, _, share = item.partition("=")
        if name in parsed or name not in bank.series_names:
            raise ValueError("objective names each series of %s at most once" % ", ".join(bank.series_names))
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
