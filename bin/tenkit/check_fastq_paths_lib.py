#!/usr/bin/env python3
#
# Copyright (c) 2016 10x Genomics, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import os
import sys


def make_parser():
    parser = argparse.ArgumentParser(
        description="Translate paths that link to pipestances or bcl2fastq paths to analysis pipeline-compatible paths."
    )
    parser.add_argument("--fastqs", help="Supplied input FASTQ path", required=True)
    parser.add_argument(
        "--project", default=None, help="Project within a bcl2fastq/mkfastq folder to extract from"
    )
    return parser


# TODO: this should go in tenkit.bcl but the martian reference needs
# to be removed before this can be invoked from outside a stage
# context
def get_bcl2fastq_output_folder(path):
    """If the path contains a pipestance or bcl2fastq output in a.

    known location, return the path to the bcl2fastq output
    root.

    :param path: The FASTQ input path.
    :param project: The supplied project field.
    :return: The FASTQ path that is legal for a downstream run
    :raises: ValueError if the path/project path is illegal
    """
    if not os.path.exists(path):
        raise ValueError(f"Invalid path: {path}")

    link = os.path.join(path, "outs", "fastq_path")
    # check path pointed to pipestance
    if os.path.exists(link):
        return os.path.realpath(link)
    # check path pointed to pipestance/outs
    link = os.path.join(path, "fastq_path")
    if os.path.exists(link):
        return os.path.realpath(link)
    # check bcl2fastq folder default mode (if --reports-dir/--stats-dir sent elsewhere this would be a false negative)
    if os.path.exists(os.path.join(path, "Reports")) or os.path.exists(os.path.join(path, "Stats")):
        return os.path.realpath(path)

    # otherwise, can't automatically detect bcl2fastq output folder hierarchy - this is None
    return None


# TODO: this should go in tenkit.bcl but the martian reference needs
# to be removed before this can be invoked from outside a stage
# context
def get_projects(bcl2fastq_path):
    dirnames = next(os.walk(bcl2fastq_path))[1]

    project_dirs = [dn for dn in dirnames if dn not in ("Reports", "Stats")]
    return [os.path.basename(project) for project in project_dirs]


def main():
    parser = make_parser()
    args = parser.parse_args()

    fastq_paths = [x for x in args.fastqs.strip(",").split(",") if len(x) > 0]
    output_paths = []

    for path in fastq_paths:
        abs_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(abs_path):
            sys.stderr.write(f"Invalid path to --fastqs: {abs_path}\n")

        try:
            fastq_path = get_bcl2fastq_output_folder(abs_path)
            # if not a recognized path, still allow the path to
            # fall through to the
            if not fastq_path:
                output_paths.append(abs_path)
                continue
            projects = get_projects(fastq_path)

            if not projects:
                # just return the path (this does the right thing for demux pipelines!)
                output_paths.append(fastq_path)
            if args.project:
                if args.project in projects:
                    output_paths.append(os.path.join(fastq_path, args.project))
                # just don't append if project doesn't match, but don't complain
            elif len(projects) > 1:
                projects_list = "\n".join(sorted(projects))
                raise ValueError(
                    "The --project argument must be specified if BCLs "
                    "were demultiplexed into multiple project folders. "
                    f"Options:\n{projects_list}"
                )
            # project length == 1
            elif len(projects) == 1:
                output_paths.append(os.path.join(fastq_path, projects[0]))

        except ValueError as ex:
            sys.stderr.write(f"{ex!s}\n")
            return 1

    # if no paths selected because --project specified, say so
    if args.project and len(fastq_paths) > 0 and len(output_paths) == 0:
        sys.stderr.write(f"Could not find any paths that matched --project value: {args.project}\n")
        return 1

    # at this point all paths containing either paths to the bcl2fastq root folder
    # or a mkfastq pipeline will have been translated to the entry point compatible
    # with tenkit.fasta find functions.  Print them back out, comma-delimited and exit.
    print(*output_paths, sep=",")
    return 0


if __name__ == "__main__":
    sys.exit(main())
