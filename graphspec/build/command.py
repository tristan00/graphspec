from graphspec.infra.timing import timed

from .builder import Builder
from .readers import declarations, namespace


@timed("build_command.script")
def script(path, sources):
    held = namespace(path, sources)
    if "builder" in held:
        if not isinstance(held["builder"], Builder):
            raise ValueError("a build script exports its Builder as builder")
        return held["builder"]
    return Builder(declarations(held), sources)
