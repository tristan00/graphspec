import hashlib
from pathlib import Path

from .runtime import output_path
from .timing import timed


def files(packages):
    return sorted({path for package in packages for path in Path(package).rglob("*.py")})


def relative(path, packages):
    root = min(
        (Path(package) for package in packages if path.is_relative_to(package)),
        key=lambda package: len(package.parts),
    )
    return path.relative_to(root.parent)


@timed("sources.entries")
def entries(packages):
    packages = tuple(Path(package) for package in packages)
    result = [(path, relative(path, packages)) for path in files(packages)]
    destinations = [destination for _, destination in result]
    if len(set(destinations)) != len(destinations):
        raise ValueError("source packages produce overlapping snapshot paths")
    return result


@timed("sources.digest")
def digest(packages):
    value = hashlib.sha256()
    for path, destination in entries(packages):
        value.update(destination.as_posix().encode())
        value.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return value.hexdigest()


@timed("sources.snapshot")
def snapshot(packages, destination):
    selected = entries(packages)
    destination = output_path(destination)
    destination.mkdir(parents=True)
    for path, relative_path in selected:
        copy = destination / relative_path
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
