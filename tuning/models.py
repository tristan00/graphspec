import logging
import time

from graphspec.spec import canonical

from .affinity import AffinityNet
from .events import EventGraphNet, MemoryNet, ProjectionNet
from .message_net import MixedNet
from .model_spec import describe, space
from .neighbors import SequenceNet, TreeNet, WalkNet
from .signal import SignalNet
from .snapshot import SnapshotNet

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
