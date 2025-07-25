#!/usr/bin/env python
#
# Copyright (c) 2018 10x Genomics, Inc. All rights reserved.

"""Truncate a text or json file.

For text files, elide all the but the first and last several
lines.

For a json file by substituting placeholders in such a way as to still be
mostly usable by things which attempt to deserialize them.
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque

MAX_LOAD_SIZE = 256 * 1024 * 1024
OUT_SIZE = 1024 * 1024

BEGIN_LINES = 100
END_LINES = 500
MAX_LINE_LEN = 160

STRING_TYPES = (str, bytes)


def _copy_start(data, dest):
    """Reads the first BEGIN_LINES from data.

    It has safety checks to prevent loading more than MAX_LOAD_SIZE bytes.
    Leaves data no more than MAX_LOAD_SIZE from the end of the file, noting the
    number of skipped bytes, if any.

    Returns:
        True if end of file was reached.
    """
    # First read the first BEGIN_LINES.  If any line is >= OUT_SIZE,
    # abort reading those lines.
    for _ in range(BEGIN_LINES):
        line = data.readline(OUT_SIZE)
        if not line:
            return True
        elif len(line) > MAX_LINE_LEN:
            dest.write(line[: MAX_LINE_LEN - 4])
            dest.write("...\n".encode())  # noqa: UP012
            if len(line) == OUT_SIZE:
                # Didn't actually find the end of the line. Stop reading lines
                # and prepare to seek ahead.
                break
        else:
            dest.write(line)
    # Find out how much is left to read.  If it's > MAX_LOAD_SIZE, seek to
    # OUT_SIZE before the end of the file and continue scanning lines from
    # there.
    start_point = data.tell()
    data.seek(0, os.SEEK_END)
    size = data.tell()
    if size <= start_point:
        return True
    if size > MAX_LOAD_SIZE and size > start_point + OUT_SIZE:
        dest.write(f"... {size - OUT_SIZE - start_point} bytes elided ...\n".encode())
        data.seek(-OUT_SIZE, os.SEEK_END)
    else:
        # Continue from where we left off.
        data.seek(start_point)
    return False


def truncate_text(data, dest):
    """Take the first BEGIN_LINES and last END_LINES lines from a file.

    Note the number of elided lines.
    """
    if _copy_start(data, dest):
        return
    buf = deque(maxlen=END_LINES)
    elided = 0
    # Safe to read entire lines because we've already checked the file size.
    for line in data:
        if len(buf) >= END_LINES:
            elided += 1
        if len(line) > MAX_LINE_LEN:
            line = line[: MAX_LINE_LEN - 4] + b"...\n"
        buf.append(line)
    if elided:
        dest.write(f"... {elided} lines elided ...\n".encode())
    for line in buf:
        dest.write(line)


MAX_STR_LEN = 16
MAX_LIST_LEN = 16


def _truncate_object(obj):
    """Remove elements from a json object to reduce its size.

    Notes the original size in such a way as to not break expected types
    for later readers.

    - Integers and booleans are left alone.
    - Strings less than _MAX_STR_LEN are left alone.  Longer strings are replaced with
      "...suffix".
    - Lists or objects of _MAX_LIST_LEN or fewer elements are truncated
      recursively.
    - For lists of more than _MAX_LIST_LEN elements, the first
      (_MAX_LIST_LEN-1) elements are truncated recursively, and the next is
      replaced by _truncate_report.
    - Objects of more than 2 elements are replaced with {"truncated": <N>}
    """
    if isinstance(obj, dict):
        if len(obj) > MAX_LIST_LEN:
            return {"truncated": len(obj)}
        return {key: _truncate_object(value) for key, value in obj.items()}
    elif isinstance(obj, STRING_TYPES):
        if len(obj) > MAX_STR_LEN:
            obj = os.path.basename(obj)
        if len(obj) > MAX_STR_LEN:
            return obj[: MAX_STR_LEN - 9] + "..." + obj[-6:]
    elif isinstance(obj, list):
        to_truncate = len(obj) - MAX_LIST_LEN
        obj = [_truncate_object(item) for item in obj[:MAX_LIST_LEN]]
        if to_truncate > 0:
            obj[-1] = _truncate_report(obj[-1], to_truncate)
    return obj


def _truncate_report(obj, size):
    """Returns the integer size wrapped to be the same type as obj.

    - "str" -> "<size>"
    - 1 -> <size>
    - {...} -> {"truncated elements": <size>}
    - empty list -> [<size>]
    - non-empty lists recurse.
    """
    if isinstance(obj, STRING_TYPES):
        return f"[{size} more]"
    elif isinstance(obj, dict):
        return {"truncated elements": size}
    elif isinstance(obj, list):
        if obj:
            return [_truncate_report(obj[0], size)]
        return [size]
    return size


def truncate_json(obj, dest):
    """Json-serialize a truncated version of obj to dest."""
    try:
        obj = _truncate_object(obj)
    except:
        obj = {"truncated": True}
    json.dump(obj, dest)


def truncate_file(filename):
    """Open the given named file and print a truncated version to stdout."""
    with open(filename, "rb") as data_binary:
        is_json = False
        data_binary.seek(0, os.SEEK_END)
        size = data_binary.tell()
        data_binary.seek(0)
        if size <= MAX_LOAD_SIZE:
            is_json = True
            try:
                data = os.fdopen(data_binary.fileno())
                obj = json.load(data)
            except:
                data_binary.seek(0)
                is_json = False
        if is_json:
            try:
                truncate_json(obj, sys.stdout)
            except UnicodeDecodeError:
                sys.stdout.write("[binary data]\n")
        else:
            data_binary.seek(0)
            with open(1, "wb") as stdout_binary:
                try:
                    truncate_text(data_binary, stdout_binary)
                except UnicodeDecodeError:
                    stdout_binary.write("[binary data]\n".encode())  # noqa UP012


if __name__ == "__main__":
    truncate_file(sys.argv[1])
