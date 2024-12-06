#!/usr/bin/env python
#
# Copyright (c) 2019 10X Genomics, Inc. All rights reserved.
#

from __future__ import annotations

import json
import os
import subprocess

import martian
import numpy as np

import tenkit.log_subprocess as tk_subproc

# Find lib/bin/tiffer by relative path.
_TIFFER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "bin",
    "tiffer",
)


def try_call_tiffer_info(image_path):
    """Run tiffer info to get all information about a file."""
    call = [_TIFFER_PATH, "info", image_path]
    with tk_subproc.Popen(call, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        (proc_stdout, proc_stderr) = proc.communicate(None)
        if proc.returncode != 0:
            msgs = [
                f"Error reading image to get image metadata from {image_path}:",
                proc_stdout.decode(),
                proc_stderr.decode(),
            ]
            raise RuntimeError("\n".join([x for x in msgs if x]))
    imageinfo = json.loads(proc_stdout)
    return imageinfo


def call_tiffer_info(image_path):
    """Run tiffer info to get all information about a file."""
    call = [_TIFFER_PATH, "info", image_path]
    with tk_subproc.Popen(call, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        (proc_stdout, proc_stderr) = proc.communicate(None)
        if proc.returncode != 0:
            msgs = [
                f"Error reading image to get image metadata from {image_path}:",
                proc_stdout.decode(),
                proc_stderr.decode(),
            ]
            martian.exit("\n".join([x for x in msgs if x]))
    imageinfo = json.loads(proc_stdout)
    return imageinfo


def call_tiffer_checksum(image_path):
    """Runs "tiffer checksum" to get a checksum for an image file."""
    call = [_TIFFER_PATH, "checksum", image_path]
    proc = tk_subproc.Popen(call, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (proc_stdout, proc_stderr) = proc.communicate(None)
    if proc.returncode != 0:
        msgs = [
            f"Error getting checksum of file {image_path}:",
            proc_stdout.decode(),
            proc_stderr.decode(),
        ]
        martian.exit("\n".join([x for x in msgs if x]))
    return proc_stdout.strip().decode()


def call_tiffer_get_num_pages(image_path):
    """Run tiffer pages to get the number of pages in the image."""
    call = [_TIFFER_PATH, "pages", image_path]
    proc = tk_subproc.Popen(call, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (proc_stdout, proc_stderr) = proc.communicate(None)
    if proc.returncode != 0:
        msgs = [
            f"Error reading image to get number of channels/pages for {image_path}:",
            proc_stdout.decode(),
            proc_stderr.decode(),
        ]
        martian.exit("\n".join([x for x in msgs if x]))
    npages = int(proc_stdout.strip())
    return npages


def call_tiffer_mem_estimate(image_path, downsample_size):
    """Run tiffer memreq with given parameters."""
    call = [_TIFFER_PATH, "memreq", image_path, "--size", str(downsample_size)]
    proc = tk_subproc.Popen(call, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (proc_stdout, proc_stderr) = proc.communicate(None)
    if proc.returncode != 0:
        msgs = [
            f"Error reading image file {image_path} while estimating memory requirements:",
            proc_stdout.decode(),
            proc_stderr.decode(),
        ]
        martian.exit("\n".join([x for x in msgs if x]))
    output_bytes = int(proc_stdout.strip())
    return output_bytes


def call_tiffer_mem_estimate_gb(image_path, downsample_size):
    """Call_tiffer_mem_estimate but in GB."""
    output_bytes = call_tiffer_mem_estimate(image_path, downsample_size)
    bytes_gb = np.ceil(output_bytes / (1024.0 * 1024.0 * 1024.0))
    return bytes_gb


def call_tiffer_resample(image_path, downsample_size, full_output_path, page=None):
    """Run tiffer with given parameters."""
    full_output_path = os.path.abspath(full_output_path)
    output_path = os.path.dirname(full_output_path)
    output_file_name = os.path.basename(full_output_path)
    (output_file_name_noext, ext) = os.path.splitext(output_file_name)
    ext = ext[1:]

    call = [
        _TIFFER_PATH,
        "resample",
        image_path,
        output_path,
        "--jsonfile",
        "--size",
        str(downsample_size),
        "--format",
        ext,
        "--outname",
        output_file_name_noext,
    ]

    if page is not None:
        call.extend(["--page", str(page)])

    try:
        _ = tk_subproc.check_output(call, stderr=subprocess.STDOUT)
        # ^ the unused returned variable is warning/error messages.
        output_json_wpath = os.path.join(output_path, f"{output_file_name_noext}.json")
        output_json = json.load(open(output_json_wpath))
    except subprocess.CalledProcessError as ex:
        martian.exit(f"Could not generate downsampled image: \n{ex.output.decode()}")
    return output_json
