import sys


def main():
    commands = {
        "synthetic": ("graphspec.synthetic", "main"),
        "build": ("tuning.build", "main"),
        "search": ("tuning.run", "main"),
        "check": ("tuning.check_models", "main"),
        "database": ("graphspec.database", "main"),
        "dashboard": ("tuning.dashboard", "main"),
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit("usage: graphspec {" + ",".join(commands) + "} [arguments]")
    import importlib

    module, function = commands[sys.argv.pop(1)]
    getattr(importlib.import_module(module), function)()


if __name__ == "__main__":
    main()
