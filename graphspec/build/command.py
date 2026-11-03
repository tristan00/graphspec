import runpy
from pathlib import Path

from graphspec.infra.timing import timed

from .builder import Builder


@timed("build_command.script")
def script(path, sources):
    namespace = runpy.run_path(str(Path(path).resolve()), init_globals=dict(sources=Path(sources).resolve()))
    if not isinstance(namespace.get("builder"), Builder):
        raise ValueError("a base-layer build script exports its Builder as builder")
    return namespace["builder"]
