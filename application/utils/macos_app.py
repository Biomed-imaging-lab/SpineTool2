"""Set native application metadata before Qt initializes Cocoa."""

import sys


def set_macos_app_name(name: str) -> None:
    if sys.platform != "darwin" or getattr(sys, "frozen", False):
        return

    import ctypes

    # Load Apple's system library to access this process's bundle metadata.
    cf = ctypes.CDLL(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )
    pointer = ctypes.c_void_p
    cf.CFBundleGetMainBundle.argtypes = []
    cf.CFBundleGetMainBundle.restype = pointer
    cf.CFBundleGetInfoDictionary.argtypes = [pointer]
    cf.CFBundleGetInfoDictionary.restype = pointer
    cf.CFStringCreateWithCString.argtypes = [pointer, ctypes.c_char_p, ctypes.c_uint32]
    cf.CFStringCreateWithCString.restype = pointer
    cf.CFDictionarySetValue.argtypes = [pointer, pointer, pointer]
    cf.CFDictionarySetValue.restype = None
    cf.CFRelease.argtypes = [pointer]
    cf.CFRelease.restype = None

    bundle = cf.CFBundleGetMainBundle()
    info = cf.CFBundleGetInfoDictionary(bundle) if bundle else None
    if not info:
        return

    # Update only this process's in-memory metadata, never Python's Info.plist.
    utf8 = 0x08000100  # Apple's kCFStringEncodingUTF8 constant.
    value = cf.CFStringCreateWithCString(None, name.encode("utf-8"), utf8)
    if not value:
        return
    try:
        for field in (b"CFBundleName", b"CFBundleDisplayName"):
            key = cf.CFStringCreateWithCString(None, field, utf8)
            if key:
                try:
                    cf.CFDictionarySetValue(info, key, value)
                finally:
                    cf.CFRelease(key)
    finally:
        cf.CFRelease(value)
