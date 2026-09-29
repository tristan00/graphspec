import hashlib
import subprocess
import sys
from pathlib import Path

import git_filter_repo as fr

root = Path('D:/graphspec').resolve()


def git(*args, data=None):
    return subprocess.check_output(['git', *args], cwd=root, input=data)


objects = {}
reader = subprocess.Popen(['git', 'cat-file', '--batch-all-objects', '--batch'], stdout=subprocess.PIPE)
while header := reader.stdout.readline():
    oid, kind, size = header.split()
    objects[oid] = (kind, reader.stdout.read(int(size)))
    reader.stdout.read(1)
reader.wait()
trees = {}
for oid, (kind, data) in objects.items():
    if kind != b'tree':
        continue
    entries = []
    while data:
        entry, data = data.split(b'\0', 1)
        entries.append((entry.split(b' ', 1)[1], data[:20].hex().encode()))
        data = data[20:]
    trees[oid] = entries
removed = set()


def subtree(oid):
    if oid in removed:
        return
    if objects[oid][0] == b'blob' and not objects[oid][1].strip():
        return
    removed.add(oid)
    for name, child in trees.get(oid, []):
        subtree(child)


for entries in trees.values():
    for name, oid in entries:
        if b'dashboard' in name.lower():
            subtree(oid)
for oid, (kind, data) in objects.items():
    if kind == b'blob' and (
        data.startswith(b'import { compact, compareValues, matchesFilter } from "./filters.mjs";')
        or (data.lstrip().lower().startswith(b'<!doctype html>') and b'/api/' in data)
    ):
        removed.add(oid)
changed = True
while changed:
    changed = False
    for oid, entries in trees.items():
        if oid not in removed and any(child in removed for name, child in entries):
            removed.add(oid)
            changed = True
unreachable = {line.split()[2] for line in git('fsck', '--full', '--unreachable', '--no-reflogs').splitlines()
               if line.startswith(b'unreachable ')}
preserved = unreachable - removed
print(f'Preexisting unreachable objects: {len(unreachable)}; dashboard objects: {len(unreachable & removed)}; unrelated objects preserved: {len(preserved)}', flush=True)
if '--audit' in sys.argv:
    raise SystemExit()
if preserved:
    pack = git('pack-objects', '.git/objects/pack/pack', data=b'\n'.join(sorted(preserved)) + b'\n').strip().decode()
    (root / '.git' / 'objects' / 'pack' / f'pack-{pack}.keep').write_text('Preserve preexisting unrelated unreachable objects\n')


def filename_callback(filename):
    return None if b'dashboard' in filename.lower() else filename


class PreserveWorkingTree(fr.RepoFilter):
    def cleanup(self, repo, repack, reset, **kwargs):
        subprocess.run(['git', 'read-tree', 'HEAD'], cwd=root, check=True)
        super().cleanup(repo, repack, False, **kwargs)


before = {}
for path in root.rglob('*'):
    if not path.is_file() or '.git' in path.parts or path.name == Path(__file__).name:
        continue
    before[path] = hashlib.sha256(path.read_bytes()).digest()
args = fr.FilteringOptions.parse_args(['--force', '--replace-refs', 'delete-no-add'])
rewrite = PreserveWorkingTree(args, filename_callback=filename_callback)
rewrite.run()
for path, digest in before.items():
    if hashlib.sha256(path.read_bytes()).digest() != digest:
        raise RuntimeError(f'Working file changed during history rewrite: {path}')
print('Working tree preserved; history rewritten.')
