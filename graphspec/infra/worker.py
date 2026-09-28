import os
import runpy
import signal
import sys
import threading


def watch_owner():
    os.read(sys.stdin.fileno(), 1)
    os.killpg(os.getpgrp(), signal.SIGKILL)


def main():
    if sys.stdin.buffer.read(1) != b"1":
        raise SystemExit("worker owner exited before startup")
    if sys.platform == "linux":
        threading.Thread(target=watch_owner, daemon=True).start()
    sys.argv = sys.argv[1:]
    runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
