import ctypes
import hashlib
import logging
import threading
import time
from ctypes import wintypes

from .locks import LockBusy

LOG = logging.getLogger(__name__)
KERNEL = ctypes.WinDLL("kernel32", use_last_error=True)
SECURITY = ctypes.WinDLL("advapi32", use_last_error=True)
HANDLE = wintypes.HANDLE
DWORD = wintypes.DWORD
POINTER = ctypes.c_void_p
TOKEN_QUERY = 0x0008
TOKEN_USER = 1
INSUFFICIENT_BUFFER = 122
MUTEX_ACCESS = 0x00100001
WAIT_OBJECT = 0
WAIT_ABANDONED = 0x80
WAIT_TIMEOUT = 0x102
WAIT_FAILED = 0xFFFFFFFF
LOCAL_OWNERS = set()
LOCAL_GUARD = threading.Lock()


class TokenUser(ctypes.Structure):
    _fields_ = [("sid", POINTER), ("attributes", DWORD)]


def function(library, name, result, *arguments):
    call = getattr(library, name)
    call.restype = result
    call.argtypes = list(arguments)
    return call


current_process = function(KERNEL, "GetCurrentProcess", HANDLE)
open_token = function(SECURITY, "OpenProcessToken", wintypes.BOOL, HANDLE, DWORD, ctypes.POINTER(HANDLE))
token_information = function(
    SECURITY,
    "GetTokenInformation",
    wintypes.BOOL,
    HANDLE,
    ctypes.c_int,
    POINTER,
    DWORD,
    ctypes.POINTER(DWORD),
)
sid_length = function(SECURITY, "GetLengthSid", DWORD, POINTER)
create_mutex = function(KERNEL, "CreateMutexExW", HANDLE, POINTER, wintypes.LPCWSTR, DWORD, DWORD)
wait_mutex = function(KERNEL, "WaitForSingleObject", DWORD, HANDLE, DWORD)
release_mutex = function(KERNEL, "ReleaseMutex", wintypes.BOOL, HANDLE)
close_handle = function(KERNEL, "CloseHandle", wintypes.BOOL, HANDLE)


def checked(value):
    if not value:
        raise ctypes.WinError(ctypes.get_last_error())
    return value


def user_identity():
    token = HANDLE()
    checked(open_token(current_process(), TOKEN_QUERY, ctypes.byref(token)))
    try:
        size = DWORD()
        result = token_information(token, TOKEN_USER, None, 0, ctypes.byref(size))
        if result or ctypes.get_last_error() != INSUFFICIENT_BUFFER:
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_string_buffer(size.value)
        checked(token_information(token, TOKEN_USER, buffer, size.value, ctypes.byref(size)))
        sid = TokenUser.from_buffer(buffer).sid
        return hashlib.sha256(ctypes.string_at(sid, checked(sid_length(sid)))).hexdigest()
    finally:
        checked(close_handle(token))


class Mutex:
    def __init__(self, name):
        identity = hashlib.sha256(name.encode("utf-8")).hexdigest()
        self.name = "Global\\graphspec.user-%s.%s" % (user_identity(), identity)
        self.handle = None
        self.claimed = False
        self.owned = False

    def __enter__(self):
        started = time.perf_counter()
        with LOCAL_GUARD:
            if self.name in LOCAL_OWNERS:
                raise LockBusy(self.name)
            LOCAL_OWNERS.add(self.name)
            self.claimed = True
        try:
            self.handle = checked(create_mutex(None, self.name, 0, MUTEX_ACCESS))
            result = wait_mutex(self.handle, 0)
            if result == WAIT_TIMEOUT:
                raise LockBusy(self.name)
            if result == WAIT_FAILED:
                raise ctypes.WinError(ctypes.get_last_error())
            if result not in (WAIT_OBJECT, WAIT_ABANDONED):
                raise RuntimeError("unexpected mutex wait result: %d" % result)
            self.owned = True
            LOG.info("mutex enter name=%s seconds=%.3f", self.name, time.perf_counter() - started)
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exception):
        self.close()

    def close(self):
        started = time.perf_counter()
        try:
            if self.owned:
                self.owned = False
                checked(release_mutex(self.handle))
        finally:
            try:
                if self.handle is not None:
                    handle, self.handle = self.handle, None
                    checked(close_handle(handle))
            finally:
                if self.claimed:
                    with LOCAL_GUARD:
                        LOCAL_OWNERS.remove(self.name)
                        self.claimed = False
        LOG.info("mutex exit name=%s seconds=%.3f", self.name, time.perf_counter() - started)
