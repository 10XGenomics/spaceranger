#!/usr/bin/env python
#
# Copyright (c) 2018 10X Genomics, Inc. All rights reserved.
#


import csv
import json
from collections import OrderedDict
from pathlib import Path
from shutil import copyfile

import cellranger.constants as cr_constants
import cellranger.spatial.data_utils as s_du
import cellranger.spatial.spatial_aggr_files as sa_files

__MRO__ = """
stage MERGE_TISSUE_FILES(
    in map[]  sample_defs,
    out csv   aggr_tissue_positions_list,
    out path  spatial,
    out json  loupe_map_json,
    src py    "rna/stages/spatial/merge_tissue_files",
"""


def main(args, outs):
    """MERGE_TISSUE_FILES creates a loupe_map dict for crconverter which lists all cloupe_files needed to aggregate.

    Copies spatial files from origin to the new spatial folder.

    input:
        args: MRO args
        outs: MRO outs
    """
    # Initiate gem id
    gem_id = 1
    loupe_map = {"loupe_map": OrderedDict()}
    sample_defs = args.sample_defs

    # Check that the first entry has a tissue_position_file. PARSE_CSV checks that
    # there are no duplicated entry in the csv
    if sample_defs[0].get(sa_files.AGG_TISSUE_POSITION_FIELD) is not None:
        aggregate_tissue_positions = True
        with open(outs.aggr_tissue_positions, "a") as out_csv:
            aggr_tissue_position_header = ",".join(s_du.TISSUE_POSITIONS_HEADER)
            out_csv.write(f"{aggr_tissue_position_header}\n")
    else:
        aggregate_tissue_positions = False

    for sample_def in sample_defs:
        if cr_constants.AGG_CLOUPE_FIELD not in sample_def:
            continue
        loupe_map["loupe_map"][str(gem_id)] = [sample_def[cr_constants.AGG_CLOUPE_FIELD], 1]
        if aggregate_tissue_positions:
            # Aggregate the tissue_positions.csv into one
            with open(outs.aggr_tissue_positions, "a") as out_csv:
                # Write the tissue_positions.csv header
                writer = csv.DictWriter(
                    out_csv,
                    fieldnames=s_du.TISSUE_POSITIONS_HEADER,
                    delimiter=",",
                    lineterminator="\n",
                )
                # if in first iteration fo the loop add the header.
                tissue_position_file = Path(sample_def[sa_files.AGG_TISSUE_POSITION_FIELD])
                with open(tissue_position_file) as f:
                    reader = csv.reader(f)
                    # If using the new tissue positions file skip the header
                    if tissue_position_file.name == "tissue_positions.csv":
                        next(reader)
                    for line in reader:
                        line[0] = line[0].rstrip("1") + str(gem_id)
                        writer.writerow(
                            {
                                s_du.TISSUE_POSITIONS_HEADER[0]: line[0],
                                s_du.TISSUE_POSITIONS_HEADER[1]: line[1],
                                s_du.TISSUE_POSITIONS_HEADER[2]: line[2],
                                s_du.TISSUE_POSITIONS_HEADER[3]: line[3],
                                s_du.TISSUE_POSITIONS_HEADER[4]: line[4],
                                s_du.TISSUE_POSITIONS_HEADER[5]: line[5],
                            }
                        )
            # Copy spatial files
            target_folder = Path(outs.spatial, sample_def[cr_constants.AGG_ID_FIELD])
            target_folder.mkdir(parents=True)

            copyfile(
                sample_def[sa_files.AGG_LOWRES_IMAGES_FIELD],
                target_folder / Path(sample_def[sa_files.AGG_LOWRES_IMAGES_FIELD]).name,
            )
            copyfile(
                sample_def[sa_files.AGG_HIRES_IMAGES_FIELD],
                target_folder / Path(sample_def[sa_files.AGG_HIRES_IMAGES_FIELD]).name,
            )
            scale_factors_json = sample_def[sa_files.AGG_SCALE_FACTORS_FIELD]
            with open(scale_factors_json) as json_in:
                temp = json.load(json_in)
                temp["gem_id"] = gem_id
            with open(
                target_folder / Path(sample_def[sa_files.AGG_SCALE_FACTORS_FIELD]).name,
                "w",
            ) as json_out:
                json_out.write(json.dumps(temp))
        gem_id += 1
    with open(outs.loupe_map, "w") as outfile:
        json.dump(loupe_map, outfile, indent=4)
