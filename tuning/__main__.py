import logging
import sys
import time

LOG = logging.getLogger(__name__)


def main():
    started = time.perf_counter()
    commands = {
        "inspect": ("graphspec.descriptions.inspect", "main"),
        "synthetic": ("graphspec.sources.synthetic", "main"),
        "build": ("tuning.build", "main"),
        "search": ("tuning.run", "main"),
        "check": ("tuning.check_models", "main"),
        "database": ("graphspec.sources.ladybug", "main"),
        "dashboard": ("tuning.dashboard", "main"),
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit("usage: graphspec {" + ",".join(commands) + "} [arguments]")
    import importlib

    module, function = commands[sys.argv.pop(1)]
    try:
        getattr(importlib.import_module(module), function)()
    finally:
        LOG.info("command exit module=%s seconds=%.3f", module, time.perf_counter() - started)


if __name__ == "__main__":
    main()
