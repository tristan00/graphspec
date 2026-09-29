import ctypes
from contextlib import ExitStack
from ctypes import wintypes

import psutil

from .timing import timed

KERNEL = ctypes.WinDLL("kernel32", use_last_error=True)
SIZE_T = ctypes.c_size_t
HANDLE = wintypes.HANDLE
DWORD = wintypes.DWORD
POINTER = ctypes.c_void_p
INFINITE = 0xFFFFFFFF
PROCESS_LIST = 3
EXTENDED_LIMITS = 9
SYNCHRONIZE = 0x00100000
QUERY_LIMITED = 0x1000
SET_QUOTA = 0x0100
TERMINATE = 0x0001
MORE_DATA = 234
PROCESS_GONE = 87
ACCESS_DENIED = 5
KILL_ON_JOB_CLOSE = 0x2000


class BasicLimits(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_int64),
        ("job_time", ctypes.c_int64),
        ("flags", DWORD),
        ("minimum_working_set", SIZE_T),
        ("maximum_working_set", SIZE_T),
        ("active_process_limit", DWORD),
        ("affinity", SIZE_T),
        ("priority", DWORD),
        ("scheduling", DWORD),
    ]


class IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "read_operations",
            "write_operations",
            "other_operations",
            "read_bytes",
            "write_bytes",
            "other_bytes",
        )
    ]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", BasicLimits),
        ("io", IoCounters),
        ("process_memory", SIZE_T),
        ("job_memory", SIZE_T),
        ("peak_process_memory", SIZE_T),
        ("peak_job_memory", SIZE_T),
    ]


class ProcessIds(ctypes.Structure):
    _fields_ = [("assigned", DWORD), ("count", DWORD), ("ids", SIZE_T * 1)]


def function(name, result, *arguments):
    call = getattr(KERNEL, name)
    call.restype = result
    call.argtypes = list(arguments)
    return call


create_job = function("CreateJobObjectW", HANDLE, POINTER, wintypes.LPCWSTR)
set_information = function("SetInformationJobObject", wintypes.BOOL, HANDLE, ctypes.c_int, POINTER, DWORD)
assign = function("AssignProcessToJobObject", wintypes.BOOL, HANDLE, HANDLE)
terminate = function("TerminateJobObject", wintypes.BOOL, HANDLE, wintypes.UINT)
terminate_process = function("TerminateProcess", wintypes.BOOL, HANDLE, wintypes.UINT)
query = function("QueryInformationJobObject", wintypes.BOOL, HANDLE, ctypes.c_int, POINTER, DWORD, POINTER)
open_process = function("OpenProcess", HANDLE, DWORD, wintypes.BOOL, DWORD)
in_job = function("IsProcessInJob", wintypes.BOOL, HANDLE, HANDLE, ctypes.POINTER(wintypes.BOOL))
wait_handle = function("WaitForSingleObject", DWORD, HANDLE, DWORD)
close_handle = function("CloseHandle", wintypes.BOOL, HANDLE)


def checked(value):
    if not value:
        raise ctypes.WinError(ctypes.get_last_error())
    return value


def close_process(handle):
    checked(close_handle(handle))


def stop_process(handle):
    if not terminate_process(handle, 1) and ctypes.get_last_error() != ACCESS_DENIED:
        raise ctypes.WinError(ctypes.get_last_error())


def join_process(handle):
    if wait_handle(handle, INFINITE) != 0:
        raise ctypes.WinError(ctypes.get_last_error())


class ProcessGroup:
    popen_options = {"creationflags": 0x08000000}

    @timed("windows_job.processgroup")
    def __init__(self):
        self.job = checked(create_job(None, None))
        self.assigned = False
        try:
            limits = ExtendedLimits()
            limits.basic.flags = KILL_ON_JOB_CLOSE
            checked(set_information(self.job, EXTENDED_LIMITS, ctypes.byref(limits), ctypes.sizeof(limits)))
        except BaseException:
            self.close()
            raise

    def member(self, handle):
        value = wintypes.BOOL()
        checked(in_job(handle, self.job, ctypes.byref(value)))
        return bool(value.value)

    @timed("windows_job.processgroup.attach")
    def attach(self, process):
        checked(assign(self.job, HANDLE(int(process._handle))))
        self.assigned = True
        for child in psutil.Process(process.pid).children(recursive=True):
            handle = checked(open_process(SET_QUOTA | TERMINATE | QUERY_LIMITED, False, child.pid))
            try:
                if not self.member(handle):
                    checked(assign(self.job, handle))
            finally:
                checked(close_handle(handle))

    def members(self):
        count = 1
        while True:
            size = ProcessIds.ids.offset + count * ctypes.sizeof(SIZE_T)
            buffer = ctypes.create_string_buffer(size)
            if query(self.job, PROCESS_LIST, buffer, size, None):
                header = ProcessIds.from_buffer(buffer)
                return list((SIZE_T * header.count).from_buffer(buffer, ProcessIds.ids.offset))
            if ctypes.get_last_error() != MORE_DATA:
                raise ctypes.WinError(ctypes.get_last_error())
            count = max(count * 2, ProcessIds.from_buffer(buffer).assigned)

    @timed("windows_job.processgroup.stop")
    def stop(self):
        if not self.assigned:
            return
        members = self.members()
        while members:
            with ExitStack() as ownership:
                handles = []
                for pid in members:
                    handle = open_process(SYNCHRONIZE | QUERY_LIMITED | TERMINATE, False, pid)
                    if not handle and ctypes.get_last_error() == PROCESS_GONE:
                        continue
                    checked(handle)
                    ownership.callback(close_process, handle)
                    if self.member(handle):
                        handles.append(handle)
                for handle in handles:
                    stop_process(handle)
                for handle in handles:
                    join_process(handle)
            members = self.members()
        checked(terminate(self.job, 1))
        self.assigned = False

    @timed("windows_job.processgroup.close")
    def close(self):
        if self.job is not None:
            checked(close_handle(self.job))
            self.job = None
