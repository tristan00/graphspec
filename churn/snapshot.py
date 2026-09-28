import calendar
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from .schema import ALIASES, DATES, selected_entities


def anniversary(value):
    year = value.year + 1
    return value.replace(year=year, day=min(value.day, calendar.monthrange(year, value.month)[1]))


def date_column(table, name):
    try:
        values = np.asarray(table[name].to_pylist(), dtype="datetime64[D]")
    except ValueError as error:
        raise ValueError(name + " must contain ISO dates") from error
    if np.isnat(values).any():
        raise ValueError(name + " must contain finite dates")
    return values


def dates(table):
    starts, stopped, expires = (date_column(table, name) for name in DATES)
    ends = np.minimum(stopped, expires)
    if np.any(ends < starts):
        raise ValueError("policy endings cannot precede effective dates")
    return starts, ends


def cancellation_history(table, ended):
    values = np.full(len(ended), np.nan)
    if not ended.any():
        return values
    if "pol_policy_cancelled_flag" not in table.column_names:
        raise ValueError("past policy endings require pol_policy_cancelled_flag")
    flags = table["pol_policy_cancelled_flag"].to_pylist()
    for index in np.flatnonzero(ended):
        flag = flags[index].strip().casefold()
        if flag not in ("true", "false"):
            raise ValueError("past policy cancellation flags must be True or False")
        values[index] = float(flag == "true")
    return values


@dataclass(frozen=True)
class Snapshot:
    columns: dict
    starts: np.ndarray
    active: np.ndarray
    ended_days: np.ndarray
    cancellations: np.ndarray
    cutoff: np.datetime64

    @classmethod
    def at(cls, table, prediction, fields):
        selected_entities(fields, table.column_names)
        starts, ends = dates(table)
        cutoff = np.datetime64(prediction, "D")
        visible = starts <= cutoff
        past = table.filter(pa.array(visible))
        if not past.num_rows:
            raise ValueError("no policies are visible at prediction time")
        starts, ends = starts[visible], ends[visible]
        ended = ends < cutoff
        for original, alias in ALIASES.items():
            if original in fields and alias in past.column_names and not past[original].equals(past[alias]):
                raise ValueError(
                    alias + " is not an alias of " + original + "; declare its identity separately"
                )
        columns = {name: past[name].to_pylist() for name in ("policy_key", "client_key", *fields)}
        days = (cutoff - np.where(ended, ends, cutoff)).astype(int)
        return cls(columns, starts, ends > cutoff, days, cancellation_history(past, ended), cutoff)


def active_clients(table, moment):
    starts, ends = dates(table)
    moment = np.datetime64(moment, "D")
    return set(table["client_key"].filter(pa.array((starts <= moment) & (ends > moment))).to_pylist())


def outcomes(table, evaluation, cohort):
    active = active_clients(table, evaluation)
    return [float(key in active) for key in cohort]
