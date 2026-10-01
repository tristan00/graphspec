import numpy as np
import torch

from graphspec.infra.timing import timed
from tuning.families.heads import FIELD, Attached


class Share:
    scaled = False
    matures = None
    columns = dict(trials="FLOAT")
    fields = ("trials",)

    @timed("shares.share.series")
    def series(self, stated, where):
        if type(stated) is not str or not stated:
            raise ValueError("%s names its one series, found %r" % (where, stated))
        return (stated,)

    @timed("shares.share.written")
    def written(self, series):
        return series[0]

    @timed("shares.share.points")
    def points(self, rows, series, where):
        share = np.asarray(rows["y"], np.float64)
        trials = np.asarray([np.nan if value is None else value for value in rows["trials"]], np.float64)
        if np.any((share < 0) | (share > 1)):
            raise ValueError("a share of %s lies outside 0 and 1" % where)
        if not np.all(trials >= 1) or len(np.unique(rows["task"])) != len(share):
            raise ValueError("%s holds one share of one or more trials for a task" % where)
        return dict(series=[series[0]] * len(share), trials=trials.tolist())

    @timed("shares.share.staged")
    def staged(self, stated, series, where):
        share, trials = stated
        return [dict(y=share, trials=trials)]

    @timed("shares.share.read")
    def read(self, rows):
        (row,) = rows
        return [row["y"], row["trials"]]

    @timed("shares.share.predicted")
    def predicted(self, values, task):
        return values


class Expected:
    reads = ("share",)
    parameters = ()

    @timed("shares.expected.checked")
    def checked(self, stated, series, where):
        return {}

    @timed("shares.expected.value")
    def value(self, stated, points):
        tasks, owner = np.unique(np.asarray(points["task"], np.int64), return_inverse=True)
        counts = np.asarray(points["y"], np.float64) * np.asarray(points["trials"], np.float64)
        return tasks, np.bincount(owner.reshape(-1), counts, minlength=len(tasks))


class ShareHead(torch.nn.Module):
    @timed("shares.sharehead")
    def __init__(self, hidden):
        super().__init__()
        self.output = torch.nn.Linear(hidden * 2 + 1, 1)

    def forward(self, h, series, b, points):
        trials = torch.log1p(b[FIELD + "trials"][points])[:, None]
        return torch.sigmoid(self.output(torch.cat((h, series, trials), -1)).squeeze(-1))


@timed("shares.shared")
def shared(hidden, spec, registry):
    return ShareHead(hidden)


class Crossed:
    reads = ("share",)

    def __call__(self, predictions, batch, bank):
        held, share, trials = predictions.clamp(1e-6, 1 - 1e-6), batch["target"], batch[FIELD + "trials"]
        return -(trials * (share * held.log() + (1 - share) * (1 - held).log())).sum() / trials.sum()


class Absolute:
    reads = ("share",)
    better = "lower"
    label, short = "Validation absolute error", "MAE"

    @timed("shares.absolute")
    def __call__(self, held):
        error = np.abs(held.prediction - held.target)
        return dict(
            value=float(error.mean()),
            series={name: float(error[held.series == code].mean()) for code, name in enumerate(held.names)},
        )


HEAD = Attached((), shared)
