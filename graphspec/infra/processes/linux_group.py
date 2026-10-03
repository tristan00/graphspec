import ctypes
import os
import signal

from graphspec.infra.timing import timed

LIBC = ctypes.CDLL(None, use_errno=True)
SET_CHILD_SUBREAPER = 36
GET_CHILD_SUBREAPER = 37


@timed("linux_group.checked")
def checked(result):
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


class ProcessGroup:
    popen_options = {"start_new_session": True}

    @timed("linux_group.processgroup")
    def __init__(self):
        self.process = None
        self.previous = ctypes.c_int()
        checked(LIBC.prctl(GET_CHILD_SUBREAPER, ctypes.byref(self.previous), 0, 0, 0))
        checked(LIBC.prctl(SET_CHILD_SUBREAPER, 1, 0, 0, 0))

    @timed("linux_group.ProcessGroup.attach")
    def attach(self, process):
        self.process = process

    @timed("linux_group.processgroup.stop")
    def stop(self):
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait()
        while True:
            try:
                os.waitpid(-self.process.pid, 0)
            except ChildProcessError:
                break
        self.process = None

    @timed("linux_group.processgroup.close")
    def close(self):
        checked(LIBC.prctl(SET_CHILD_SUBREAPER, self.previous.value, 0, 0, 0))
