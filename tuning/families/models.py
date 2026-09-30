import logging
import time

from graphspec.operations.parameters import canonical

from .affinity import AffinityNet
from .event_graph import EventGraphNet
from .memory import MemoryNet
from .mixed import MixedNet
from .model_spec import describe, space
from .projection import ProjectionNet
from .sequence import SequenceNet
from .signal import SignalNet
from .snapshot import SnapshotNet
from .tree import TreeNet
from .walks import WalkNet

LOG = logging.getLogger(__name__)
NETS = dict(
    snapshot=SnapshotNet,
    mixed=MixedNet,
    event_graph=EventGraphNet,
    memory=MemoryNet,
    projection=ProjectionNet,
    sequence=SequenceNet,
    tree=TreeNet,
    walks=WalkNet,
    affinity=AffinityNet,
    signal=SignalNet,
)


def build(cfg, registry):
    started = time.perf_counter()
    spec = canonical(space(registry), cfg["model"])
    cfg = dict(cfg, model=spec)
    net = NETS[spec["view"]](cfg, registry)
    LOG.info(
        "build exit seconds=%.4f parameters=%d model=%s",
        time.perf_counter() - started,
        sum(p.numel() for p in net.parameters()),
        describe(spec),
    )
    return net
