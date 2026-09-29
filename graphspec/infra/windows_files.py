import ctypes
import logging
import msvcrt
import os
import time
from ctypes import wintypes

LOG = logging.getLogger(__name__)

KERNEL = ctypes.WinDLL("kernel32", use_last_error=True)
GENERIC_READ = 0x80000000
DELETE = 0x00010000
SHARE_READ_WRITE_DELETE = 0x00000007
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x00000080
INVALID_HANDLE = ctypes.c_void_p(-1).value
FILE_RENAME_INFO_EX = 22
REPLACE_EXISTING_POSIX = 0x00000003

KERNEL.CreateFileW.argtypes = (
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
)
KERNEL.CreateFileW.restype = wintypes.HANDLE
KERNEL.CloseHandle.argtypes = (wintypes.HANDLE,)
KERNEL.CloseHandle.restype = wintypes.BOOL
KERNEL.SetFileInformationByHandle.argtypes = (
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.LPVOID,
    wintypes.DWORD,
)
KERNEL.SetFileInformationByHandle.restype = wintypes.BOOL


class RenameInfo(ctypes.Structure):
    _fields_ = (
        ("flags", wintypes.DWORD),
        ("root", wintypes.HANDLE),
        ("name_length", wintypes.DWORD),
        ("name", wintypes.WCHAR * 1),
    )


def replace(source, destination):
    started = time.perf_counter()
    encoded = os.fspath(destination).encode("utf-16-le")
    size = RenameInfo.name.offset + len(encoded) + ctypes.sizeof(wintypes.WCHAR)
    buffer = ctypes.create_string_buffer(max(ctypes.sizeof(RenameInfo), size))
    record = RenameInfo.from_buffer(buffer)
    record.flags, record.name_length = REPLACE_EXISTING_POSIX, len(encoded)
    ctypes.memmove(ctypes.addressof(buffer) + RenameInfo.name.offset, encoded, len(encoded))
    handle = KERNEL.CreateFileW(
        os.fspath(source), DELETE, SHARE_READ_WRITE_DELETE, None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None
    )
    if handle == INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not KERNEL.SetFileInformationByHandle(handle, FILE_RENAME_INFO_EX, buffer, len(buffer)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        KERNEL.CloseHandle(handle)
        LOG.debug("replace exit seconds=%.4f", time.perf_counter() - started)


def open_read(path):
    started = time.perf_counter()
    handle = KERNEL.CreateFileW(
        os.fspath(path),
        GENERIC_READ,
        SHARE_READ_WRITE_DELETE,
        None,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        None,
    )
    if handle == INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        KERNEL.CloseHandle(handle)
        raise
    try:
        return os.fdopen(descriptor, "r", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise
    finally:
        LOG.debug("open_read exit seconds=%.4f", time.perf_counter() - started)
