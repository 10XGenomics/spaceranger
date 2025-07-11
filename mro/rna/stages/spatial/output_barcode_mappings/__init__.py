# Copyright (c) 2025 10x Genomics, Inc. All rights reserved.
"""Output 2 um barcode name, its corresponding binned barcode names and cell ID to parquet."""

from collections import OrderedDict

import martian
import numpy as np
import pyarrow as pa

import cellranger.fast_utils as fast_utils  # pylint: disable=no-name-in-module,unused-import
from cellranger.parquet_io import ParquetBatchWriter
from cellranger.spatial.hd_feature_slice import (
    CELL_SEGMENTATION_MASK_NAME,
    NUCLEUS_SEGMENTATION_MASK_NAME,
    SEGMENTATION_GROUP_NAME,
    HdFeatureSliceIo,
    bin_name_from_bin_size_um,
)

__MRO__ = """
stage OUTPUT_BARCODE_MAPPINGS(
    in h5 hd_feature_slice,
    in  map<BinLevelInfo> bin_infos,
    out parquet barcode_mappings,
    src py "stages/spatial/output_barcode_mappings",
) using (
    mem_gb   = 4,
    volatile = strict,
)
"""

IS_NUCLEUS_FIELD_NAME = "in_nucleus"
IS_CELL_FIELD_NAME = "in_cell"
CELL_ID_FIELD_NAME = "cell_id"


def main(args, outs):

    if not args.hd_feature_slice:
        martian.clear(outs)
        return

    with HdFeatureSliceIo(args.hd_feature_slice) as feature_slice:
        cell_segmentation_mask_dataset = f"{SEGMENTATION_GROUP_NAME}/{CELL_SEGMENTATION_MASK_NAME}"
        segmentation_mask_dataset = f"{SEGMENTATION_GROUP_NAME}/{NUCLEUS_SEGMENTATION_MASK_NAME}"

        segmentation_mask = None
        cell_segmentation_mask = None
        if segmentation_mask_dataset in feature_slice.h5_file:
            segmentation_mask = feature_slice.load_counts_from_group_name(
                segmentation_mask_dataset, 1
            )
        if cell_segmentation_mask_dataset in feature_slice.h5_file:
            cell_segmentation_mask = feature_slice.load_counts_from_group_name(
                cell_segmentation_mask_dataset, 1
            )

        sorted_bin_scales = sorted(x["scale"] for x in args.bin_infos.values())

        # Get dictionary of data that should be written to parquet file
        write_barcode_data_to_parquet_file(
            nrows=feature_slice.nrows(),
            ncols=feature_slice.ncols(),
            sorted_bin_scales=sorted_bin_scales,
            spot_pitch=int(feature_slice.metadata.spot_pitch),
            output_file_path=outs.barcode_mappings,
            segmentation_mask=segmentation_mask,
            cell_segmentation_mask=cell_segmentation_mask,
        )


def bin_scale_to_name(bin_scale: int, spot_pitch: int) -> str:
    return bin_name_from_bin_size_um(bin_scale * spot_pitch)


def write_barcode_data_to_parquet_file(
    nrows: int,
    ncols: int,
    sorted_bin_scales: list[int],
    spot_pitch: int,
    output_file_path: str,
    segmentation_mask: np.ndarray | None = None,
    cell_segmentation_mask: np.ndarray | None = None,
) -> None:
    """Writes barcode data for each grid location in the feature slice to a Parquet file.

    This function generates barcodes based on bin information and spot pitch, and includes
    segmentation data (such as cell IDs and nucleus flags) if segmentation masks are provided.
    It processes each grid location and writes the corresponding barcode and segmentation data
    (nucleus and cell identification) into the output Parquet file.

    Args:
        nrows: The number of rows in the feature slice grid.
        ncols: The number of columns in the feature slice grid.
        sorted_bin_scales: A list of bin scales to output. Expected to be in sorted order
        spot_pitch: The spot pitch in micrometers, used to calculate the scaled row and column
                    positions for barcode generation.
        output_file_path: path to the output file.
        segmentation_mask: An optional 2D numpy array (shape: nrows x ncols) containing segmentation
                            data for the nucleus, where non-zero values indicate the presence of
                            a nucleus. If not provided, no nucleus segmentation will be included.
        cell_segmentation_mask: An optional 2D numpy array (shape: nrows x ncols) containing
                                     expanded cell segmentation data, where non-zero values indicate
                                     the presence of a cell. If not provided, no cell segmentation
                                     will be included.

    Returns:
        None: The function directly writes the generated barcode data to the output Parquet file.
    """
    # Sort bin infos by scale to make sure we have 2um, 8um, 16um, 48um, ... in order
    sorted_bin_names = OrderedDict(
        (x, bin_scale_to_name(bin_scale=x, spot_pitch=spot_pitch)) for x in sorted_bin_scales
    )

    # Check if masks are provided once before the loop
    has_segmentation_mask = segmentation_mask is not None
    has_cell_segmentation_mask = cell_segmentation_mask is not None

    # the fields for our records
    fields = [pa.field(bin_name, pa.string()) for bin_name in sorted_bin_names.values()]
    if has_segmentation_mask or has_cell_segmentation_mask:
        fields.append(
            pa.field(CELL_ID_FIELD_NAME, pa.string(), nullable=True)
        )  # mark field as nullable
    if has_segmentation_mask:
        fields.append(pa.field(IS_NUCLEUS_FIELD_NAME, pa.bool_()))
    if has_cell_segmentation_mask:
        fields.append(pa.field(IS_CELL_FIELD_NAME, pa.bool_()))

    # the schema for our records
    schema = pa.schema(fields)

    with ParquetBatchWriter(output_file_path, schema=schema) as parquet_writer:
        # Iterate through each grid location
        for r in range(nrows):
            for c in range(ncols):

                record = {}
                # Generate barcodes for each bin size
                for bin_scale, bin_name in sorted_bin_names.items():
                    # Generate barcode string for this bin size
                    barcode_str = fast_utils.SquareBinIndex(
                        row=r // bin_scale, col=c // bin_scale, size_um=bin_scale * spot_pitch
                    ).with_default_gem_group()

                    record[bin_name] = barcode_str

                if has_segmentation_mask:
                    record[IS_NUCLEUS_FIELD_NAME] = bool(segmentation_mask[r, c])
                if has_cell_segmentation_mask:
                    record[IS_CELL_FIELD_NAME] = bool(cell_segmentation_mask[r, c])

                if has_cell_segmentation_mask or has_segmentation_mask:
                    record[CELL_ID_FIELD_NAME] = None
                    if has_cell_segmentation_mask and (cell_id := cell_segmentation_mask[r, c]):
                        record[CELL_ID_FIELD_NAME] = fast_utils.CellId(
                            id=cell_id
                        ).with_default_gem_group()
                    elif has_segmentation_mask and (cell_id := segmentation_mask[r, c]):
                        record[CELL_ID_FIELD_NAME] = fast_utils.CellId(
                            id=cell_id
                        ).with_default_gem_group()

                parquet_writer.add_record(record=record)
