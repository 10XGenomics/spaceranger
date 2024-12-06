# Copyright (c) 2023 10X Genomics, Inc. All rights reserved.
"""HD Feature slice matrix where we store each feature as a 2D matrix in the spatial grid."""

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum

import h5py as h5
import numpy as np
from scipy.sparse import coo_matrix

import cellranger.cell_typing.cas_postprocessing as cas
import cellranger.h5_constants as h5_constants
import cellranger.matrix as cr_matrix
import cellranger.spatial.image_util as image_util
from cellranger.fast_utils import (  # pylint: disable=no-name-in-module,unused-import
    SquareBinIndex,
)
from cellranger.feature_ref import FeatureReference
from cellranger.spatial.slide_design_o3 import (  # pylint: disable=no-name-in-module, import-error
    VisiumHdLayout,
    VisiumHdSlideWrapper,
)
from cellranger.spatial.transform import normalize_perspective_transform
from cellranger.spatial.visium_hd_schema_pb2 import (  # pylint: disable=no-name-in-module,import-error
    GridIndex2D,
)

# Dataset names of the COO matix components
ROW_DATASET_NAME = "row"
COL_DATASET_NAME = "col"
DATA_DATASET_NAME = "data"

METADATA_JSON_ATTR_NAME = "metadata_json"
UMIS_GROUP_NAME = "umis"
TOTAL_UMIS_GROUP_NAME = "total"
FEATURE_SLICES_GROUP_NAME = "feature_slices"
FEATURE_SLICE_H5_FILETYPE = "feature_slice"
VERSION_ATTR_NAME = "version"
FEATURE_SLICE_H5_VERSION = "0.1.0"
READS_GROUP_NAME = "reads"

MASKS_GROUP_NAME = "masks"
FILTERED_SPOTS_GROUP_NAME = "filtered"

IMAGES_GROUP_NAME = "images"
MICROSCOPE_IMAGE_GROUP_NAME = "microscope"
CYTASSIST_IMAGE_GROUP_NAME = "cytassist"

SECONDARY_ANALYSIS_NAME = "secondary_analysis"
CLUSTERING_NAME = "clustering"
BINNING_PREFIX = "square"
MICROMETERS = "um"
GENE_EXPRESSION_CLUSTERING = "gene_expression"
GRAPHCLUSTERING_NAME = "graphclust"

CELL_ANNOTATIONS_GROUP_NAME = "cell_annotations"
CELL_TYPE_IDX_DATASET_NAME = "cell_type_idx"
CELL_ANNOTATIONS_MATRIX_NAME = "matrix"
OUT_OF_TISSUE_SENTINEL = "out-of-tissue"


class MatrixDataKind(Enum):
    """Data stored as a sparse matrix falls in one of these kinds."""

    RAW_COUNTS = "raw_counts"
    CATEGORICAL = "categorical"
    BINARY_MASK = "binary_mask"
    GRAY_IMAGE_U8 = "gray_image_u8"
    FLOAT = "float"


class MatrixDataDomain(Enum):
    """Data stored as a sparse matrix is defined in this domain."""

    # The data is defined over all the spots
    ALL_SPOTS = "all_spots"
    # The data is defined over only the spots that are filtered (cells / spots under tissue)
    FILTERED_SPOTS = "filtered_spots"


@dataclass
class HdFeatureSliceMatrixMetadata:
    """Metadata associated with HD feature slice matrix."""

    kind: str
    domain: str
    binning_scale: int

    @classmethod
    def new(
        cls,
        kind: MatrixDataKind,
        domain: MatrixDataDomain,
        binning_scale: int,
    ):
        """Create a new HdFeatureSliceMatrixMetadata object."""
        return cls(kind=kind.value, domain=domain.value, binning_scale=binning_scale)

    @classmethod
    def default(cls):
        return cls.new(MatrixDataKind.RAW_COUNTS, MatrixDataDomain.ALL_SPOTS, binning_scale=1)


class CooMatrix:
    """Sparse matrix in COO format with convenience functions for read[writing] from[to] H5."""

    row: list[int]
    col: list[int]
    data: list[int | float]

    def __init__(
        self,
        row: list[int] | None = None,
        col: list[int] | None = None,
        data: list[int | float] | None = None,
    ):
        row = row if row is not None else []
        col = col if col is not None else []
        data = data if data is not None else []
        assert len(row) == len(col)
        assert len(row) == len(data)
        self.row = row
        self.col = col
        self.data = data

    def insert(self, grid_index, data: int | float):
        """Insert an entry into the matrix."""
        if data != 0:
            self.row.append(grid_index.row)
            self.col.append(grid_index.col)
            self.data.append(data)

    def to_hdf5(self, group: h5.Group, metadata: HdFeatureSliceMatrixMetadata):
        """Write the matrix to a h5 group."""
        group.attrs[METADATA_JSON_ATTR_NAME] = json.dumps(asdict(metadata))
        for name, arr in [
            (ROW_DATASET_NAME, self.row),
            (COL_DATASET_NAME, self.col),
            (DATA_DATASET_NAME, self.data),
        ]:
            group.create_dataset(
                name,
                data=np.array(arr),
                chunks=(cr_matrix.HDF5_CHUNK_SIZE,),
                maxshape=(None,),
                compression=cr_matrix.HDF5_COMPRESSION,
                shuffle=True,
            )

    @classmethod
    def from_hdf5(cls, group):
        return cls(
            row=group[ROW_DATASET_NAME][:],
            col=group[COL_DATASET_NAME][:],
            data=group[DATA_DATASET_NAME][:],
        )

    def to_coo_matrix(self, nrows, ncols):
        return coo_matrix((self.data, (self.row, self.col)), shape=(nrows, ncols))

    def to_ndarray(self, nrows, ncols, binning_scale: int = 1) -> np.ndarray:
        """Convert the COO matrix representation to a dense ndarray at the specified binning scale."""
        ncols_binned = int(np.ceil(ncols / binning_scale))
        nrows_binned = int(np.ceil(nrows / binning_scale))

        result = np.zeros((nrows_binned, ncols_binned), dtype="int32")
        for row, col, data in zip(self.row, self.col, self.data):
            result[row // binning_scale, col // binning_scale] += data
        return result


@dataclass
class TransformMatrices:
    """Various HD transform matrices."""

    spot_colrow_to_microscope_colrow: list[list[float]] | None = field(default=None)
    microscope_colrow_to_spot_colrow: list[list[float]] | None = field(default=None)
    spot_colrow_to_cytassist_colrow: list[list[float]] | None = field(default=None)
    cytassist_colrow_to_spot_colrow: list[list[float]] | None = field(default=None)

    @classmethod
    def load(cls, transform_matrices_json: str):
        with open(transform_matrices_json) as f:
            return cls(**json.load(f))

    def set_spot_to_cytassist_transform(self, transform: np.ndarray):
        self.spot_colrow_to_cytassist_colrow = transform.tolist()
        self.cytassist_colrow_to_spot_colrow = normalize_perspective_transform(
            np.linalg.inv(transform)
        ).tolist()

    def set_spot_to_microscope_transform(self, transform: np.ndarray):
        self.spot_colrow_to_microscope_colrow = transform.tolist()
        self.microscope_colrow_to_spot_colrow = normalize_perspective_transform(
            np.linalg.inv(transform)
        ).tolist()

    def get_spot_colrow_to_microscope_colrow_transform(self) -> np.ndarray | None:
        """Query spot colrow to uscope colrow transform."""
        if self.spot_colrow_to_microscope_colrow is None:
            return None
        else:
            spot_colrow_to_microscope_colrow_matrix = np.array(
                self.spot_colrow_to_microscope_colrow
            )
            # A transform can be stored either as a vec of 3d vec or a 9d vec.
            # if it is the latter convert to the a matrix.
            if spot_colrow_to_microscope_colrow_matrix.ndim == 1:
                spot_colrow_to_microscope_colrow_matrix = (
                    spot_colrow_to_microscope_colrow_matrix.reshape(3, 3)
                )
            return spot_colrow_to_microscope_colrow_matrix

    def get_spot_colrow_to_cytassist_colrow_transform(self) -> np.ndarray | None:
        """Query spot colrow to cytassist colrow transform."""
        if self.spot_colrow_to_cytassist_colrow is None:
            return None
        else:
            spot_colrow_to_cytassist_colrow = np.array(self.spot_colrow_to_cytassist_colrow)
            # A transform can be stored either as a vec of 3d vec or a 9d vec.
            # if it is the latter convert to the a matrix.
            if spot_colrow_to_cytassist_colrow.ndim == 1:
                spot_colrow_to_cytassist_colrow = spot_colrow_to_cytassist_colrow.reshape(3, 3)
            return spot_colrow_to_cytassist_colrow

    def get_spot_colrow_to_tissue_image_colrow_transform(self) -> np.ndarray:
        """Returns the spot colrow to tissue image colrow transform.

        Returns spot to uscope image transform if uscope image exists. If it does not
        returns spot to cytassist image transform. If both transforms do not exist,
        raises a ValueError.
        """
        spot_colrow_to_tissue_image = self.get_spot_colrow_to_microscope_colrow_transform()

        # If no transform to microscope image is found, we're operating
        # without a microscope image and tissue image is cytassist image
        # Thus try the cytassist tranform, and if even that is not found raise
        # a ValueError.
        if spot_colrow_to_tissue_image is None:
            spot_colrow_to_tissue_image = self.get_spot_colrow_to_cytassist_colrow_transform()

        if spot_colrow_to_tissue_image is None:
            raise ValueError(
                "Transform corresponding to neither the tissue image nor the cytassist image found in the feature slice matrix."
            )
        return spot_colrow_to_tissue_image

    def save_json(self, fname: str):
        with open(fname, "w") as f:
            json.dump(asdict(self), f)


@dataclass
class HdFeatureSliceMetadata:
    """Metadata associated with HD feature slice matrix."""

    sample_id: str | None
    sample_desc: str | None
    slide_name: str
    nrows: int
    ncols: int
    spot_pitch: float
    hd_layout_json: str | None = field(default=None)
    transform_matrices: TransformMatrices | None = field(default=None)

    @classmethod
    def new(
        cls,
        sample_id: str | None,
        sample_desc: str | None,
        slide: VisiumHdSlideWrapper,
        transform_matrices_json: str | None,
    ):
        """Create a new HdFeatureSliceMetadata object."""
        grid_size = slide.grid_size()
        if transform_matrices_json and os.path.exists(transform_matrices_json):
            transform_matrices = TransformMatrices.load(transform_matrices_json)
        else:
            transform_matrices = None
        return cls(
            sample_id=sample_id,
            sample_desc=sample_desc,
            slide_name=slide.name(),
            nrows=grid_size.row,
            ncols=grid_size.col,
            spot_pitch=slide.spot_pitch(),
            hd_layout_json=slide.layout_str(),
            transform_matrices=transform_matrices,
        )

    def hd_layout(self) -> VisiumHdLayout | None:
        if self.hd_layout_json is None:
            return None
        return VisiumHdLayout.from_json_str(self.hd_layout_json)


def compute_grid_index_of_barcode(matrix: cr_matrix.CountMatrix, slide: VisiumHdSlideWrapper):
    """Find the row/col of each barcode in the matrix.

    Returns a list of grid indices in the same order as the barcodes in the matrix
    """
    assert slide.has_two_part_barcode()
    grid_index_of_barcode = []

    for barcode in matrix.bcs:
        barcode = SquareBinIndex(barcode=barcode.decode())
        grid_index_of_barcode.append(GridIndex2D(row=barcode.row, col=barcode.col))

    return grid_index_of_barcode


def _compute_total_umis_slice(matrix: cr_matrix.CountMatrix, grid_index_of_barcode):
    """Total UMI sparse matrix."""
    umi_slice = CooMatrix()
    for bc_idx, counts in enumerate(matrix.get_counts_per_bc()):
        grid_index = grid_index_of_barcode[bc_idx]
        umi_slice.insert(grid_index, counts)
    return umi_slice


def compute_total_umis(
    matrix: cr_matrix.CountMatrix, slide: VisiumHdSlideWrapper, binning_scale: int = 1
) -> np.ndarray:
    """Total UMIs as an ndarray."""
    grid_index_of_barcode = compute_grid_index_of_barcode(matrix, slide)
    grid_size = slide.grid_size()
    return _compute_total_umis_slice(matrix, grid_index_of_barcode).to_ndarray(
        nrows=grid_size.row, ncols=grid_size.col, binning_scale=binning_scale
    )


def bin_size_um_from_bin_name(bin_name):
    """Get the binning scale from the bin name."""
    return int(bin_name.removeprefix("square_").removesuffix("um"))


class HdFeatureSliceIo:
    """Class for dealing with the HD feature slice H5.

    Where we store an "image" for each feature
    as a sparse matrix.
    """

    h5_file: h5.File
    metadata: HdFeatureSliceMetadata
    feature_ref: FeatureReference
    open_mode: str

    def __init__(self, h5_path: str, open_mode: str = "r") -> None:
        self.open_mode = open_mode
        self.h5_file = h5.File(h5_path, open_mode)
        metadata_str = self.h5_file.attrs[METADATA_JSON_ATTR_NAME]
        metadata_dict = json.loads(metadata_str)
        transform_dict = metadata_dict.pop("transform_matrices", None)
        if transform_dict is not None:
            transform_matrices = TransformMatrices(**transform_dict)
        else:
            transform_matrices = None
        self.metadata = HdFeatureSliceMetadata(
            transform_matrices=transform_matrices, **metadata_dict
        )
        self.feature_ref = FeatureReference.from_hdf5(
            self.h5_file[h5_constants.H5_FEATURE_REF_ATTR]
        )

    def __del__(self):
        self.h5_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.h5_file:
            self.h5_file.close()

    def ncols(self):
        return self.metadata.ncols

    def nrows(self):
        return self.metadata.nrows

    def slide(self) -> VisiumHdSlideWrapper:
        return VisiumHdSlideWrapper(self.metadata.slide_name, layout=self.metadata.hd_layout())

    def _get_feature_indices_from_names(self, gene_names):
        return sorted(
            [
                index
                for index, name in enumerate(self.feature_ref.get_feature_names())
                if name in gene_names
            ]
        )

    def read_feature_slices_from_gene_names(self, gene_names: set[str]) -> np.ndarray:
        """Read the slices corresponding to a set of gene names.

        Args:
            gene_names (set[str]): set of gene names to read

        Raises:
            ValueError: if all the genes requested are not in feature reference

        Returns:
            np.ndarray: an ndarray corresponding to the genes in gene_names
        """
        genes_in_matrix = set(self.feature_ref.get_feature_names())
        if not gene_names.issubset(genes_in_matrix):
            genes_not_found = ", ".join(list(gene_names - genes_in_matrix))
            raise ValueError(f"Genes {genes_not_found} asked for but are not in feature reference")
        gene_indices = self._get_feature_indices_from_names(gene_names)
        cumulative_gene_matrix = coo_matrix((self.nrows(), self.ncols()))
        feature_slices_group = self.h5_file[FEATURE_SLICES_GROUP_NAME]
        for gene_index in gene_indices:
            if str(gene_index) in feature_slices_group:
                cumulative_gene_matrix += CooMatrix.from_hdf5(
                    feature_slices_group[str(gene_index)]
                ).to_coo_matrix(self.nrows(), self.ncols())
        return cumulative_gene_matrix.todense().A

    def load_counts_from_group_name(self, group_name, binning_scale):
        """Given a group name that stores counts, load it at the given bin scale."""
        return CooMatrix.from_hdf5(self.h5_file[group_name]).to_ndarray(
            nrows=self.nrows(), ncols=self.ncols(), binning_scale=binning_scale
        )

    def _load_total_umis(self):
        return CooMatrix.from_hdf5(self.h5_file[UMIS_GROUP_NAME][TOTAL_UMIS_GROUP_NAME])

    def total_umis(self, binning_scale: int = 1):
        return self._load_total_umis().to_ndarray(
            nrows=self.nrows(), ncols=self.ncols(), binning_scale=binning_scale
        )

    def _load_clustering(
        self,
        binning_scale: int = 4,
        feature_used_to_cluster: str = GENE_EXPRESSION_CLUSTERING,
        clustering_method: str = GRAPHCLUSTERING_NAME,
    ) -> CooMatrix:
        """Read clustering from the slice into a sparse matrix.

        Args:
            binning_scale (int, optional): Binning scale of clustering.
                Accepts values 2, 4, 25, 50. Defaults to 4.
            feature_used_to_cluster (str, optional): Features using which cluster is done.
                Defaults to GENE_EXPRESSION_CLUSTERING.
            clustering_method (str, optional): Clustering method.
                Defaults to GRAPHCLUSTERING_NAME.

        Raises:
            ValueError: If the clustering at the bin level and using
                the feature and method requested is not found in data

        Returns:
            CooMatrix: Sparse matrix sith clusters assigne to all spots.
                Spots outside tissue are annotated as cluster 0.
        """
        binned_name = (
            f"{BINNING_PREFIX}_{self.metadata.spot_pitch*binning_scale:03.0f}{MICROMETERS}"
        )
        clustering_group_name = f"{binned_name}_{feature_used_to_cluster}_{clustering_method}"
        if (
            f"{SECONDARY_ANALYSIS_NAME}/{CLUSTERING_NAME}/{clustering_group_name}"
            not in self.h5_file
        ):
            raise ValueError(
                f"Feature Slice does not contain clustering group \
                {SECONDARY_ANALYSIS_NAME}/{CLUSTERING_NAME}/{clustering_group_name}. \
                Perhaps, check the clustering method and binning scale."
            )
        return CooMatrix.from_hdf5(
            self.h5_file[SECONDARY_ANALYSIS_NAME][CLUSTERING_NAME][clustering_group_name]
        )

    def _load_group_name(self, group_name: str):
        if group_name not in self.h5_file:
            # throw key error
            raise KeyError(f"Group {group_name} not found in feature slice")
        group = self.h5_file[group_name]
        metadata = HdFeatureSliceMatrixMetadata(**json.loads(group.attrs[METADATA_JSON_ATTR_NAME]))
        ncols_binned = int(np.ceil(self.ncols() / metadata.binning_scale))
        nrows_binned = int(np.ceil(self.nrows() / metadata.binning_scale))
        return (
            CooMatrix.from_hdf5(group)
            .to_coo_matrix(nrows=nrows_binned, ncols=ncols_binned)
            .toarray()
        )

    def get_umap(
        self, binning_scale: int = 4, feature_used_to_cluster: str = GENE_EXPRESSION_CLUSTERING
    ):
        """Read UMAP x, y coordinates as two numpy arrays."""
        binned_name = (
            f"{BINNING_PREFIX}_{self.metadata.spot_pitch*binning_scale:03.0f}{MICROMETERS}"
        )
        umap_group_name = (
            f"{SECONDARY_ANALYSIS_NAME}/umap/{binned_name}_{feature_used_to_cluster}_2"
        )
        umap_x = self._load_group_name(umap_group_name + "/0")
        umap_y = self._load_group_name(umap_group_name + "/1")
        return umap_x, umap_y

    def get_clustering(
        self,
        binning_scale: int = 4,
        feature_used_to_cluster: str = GENE_EXPRESSION_CLUSTERING,
        clustering_method: str = GRAPHCLUSTERING_NAME,
    ) -> np.ndarray:
        """Read clustering from the slice.

        Args:
            binning_scale (int, optional): Binning scale of clustering.
                Accepts values 2, 4, 25, 50. Defaults to 4.
            feature_used_to_cluster (str, optional): Features using which cluster is done.
                Defaults to GENE_EXPRESSION_CLUSTERING.
            clustering_method (str, optional): Clustering method.
                Defaults to GRAPHCLUSTERING_NAME.

        Returns:
            np.ndarray: Dense matrix with clusters of all spots.
                Spots outside tissue are annotated as cluster 0.
        """
        ncols_binned = int(np.ceil(self.ncols() / binning_scale))
        nrows_binned = int(np.ceil(self.nrows() / binning_scale))
        return self._load_clustering(
            binning_scale,
            feature_used_to_cluster,
            clustering_method,
        ).to_ndarray(nrows=nrows_binned, ncols=ncols_binned)

    def _apply_perspective_transform(
        self, x: int | float, y: int | float, mat: np.ndarray
    ) -> tuple[float, float]:
        """Apply perspective transform to an (x,y)-coordinate.

        Args:
            x (int | float): X-co-ordinate
            y (int | float): Y-co-ordinate
            mat (np.ndarray): Transform matrix

        Returns:
            tuple[float, float]: Transformed co-ordinates
        """
        raw = mat.dot(np.array([x, y, 1]))
        return (
            raw[0] * self.metadata.spot_pitch / raw[2],
            raw[1] * self.metadata.spot_pitch / raw[2],
        )

    def transform_from_microscope_to_spot_co_ordinates(
        self, segmentation: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Transform a polygon from microscope co-ordinates to spot co-ordinates.

        Args:
            segmentation (list[tuple[float, float]]): Segmentation
                represented as a list of tuples

        Returns:
            list[tuple[float, float]]: Transformed segmentation.
        """
        transform_matrix = np.array(
            self.metadata.transform_matrices.microscope_colrow_to_spot_colrow
        )
        return [self._apply_perspective_transform(*x, transform_matrix) for x in segmentation]

    @staticmethod
    def _create_feature_slices(matrix: cr_matrix.CountMatrix, grid_index_of_barcode):
        num_feat, _ = matrix.m.shape
        matrix.tocoo()

        feature_slices = [CooMatrix() for _ in range(num_feat)]

        for feature_idx, bc_idx, umi_count in zip(matrix.m.row, matrix.m.col, matrix.m.data):
            grid_index = grid_index_of_barcode[bc_idx]
            feature_slices[feature_idx].insert(grid_index, umi_count)

        return feature_slices

    @staticmethod
    def _write_read_group(group, grid_index_of_barcode, barcode_summary_h5: str):
        barcode_summary_datasets = {
            "sequenced_reads": "sequenced",
            "barcode_corrected_sequenced_reads": "barcode_corrected_sequenced",
            "unmapped_reads": "unmapped_reads",
            "_multi_transcriptome_split_mapped_barcoded_reads": "split_mapped",
            "_multi_transcriptome_half_mapped_barcoded_reads": "half_mapped",
        }
        with h5.File(barcode_summary_h5, "r") as f:
            for key, group_name in barcode_summary_datasets.items():
                if key not in f:
                    continue
                read_slice = CooMatrix()
                for grid_index, data in zip(grid_index_of_barcode, f[key][:]):
                    read_slice.insert(grid_index, data)
                read_slice_group = group.create_group(group_name)
                read_slice.to_hdf5(
                    read_slice_group,
                    HdFeatureSliceMatrixMetadata.default(),
                )

    @staticmethod
    def _set_metadata(h5_file, metadata: HdFeatureSliceMetadata):
        h5_file.attrs[METADATA_JSON_ATTR_NAME] = json.dumps(asdict(metadata))

    @staticmethod
    def create_h5(
        h5_path: str,
        sample_id: str | None,
        sample_desc: str | None,
        matrix: cr_matrix.CountMatrix,
        slide: VisiumHdSlideWrapper,
        barcode_summary_h5: str | None = None,
        transform_matrices_json: str | None = None,
    ) -> None:
        """Create the H5 matrix from the raw matrix and slide design."""
        assert (
            slide.has_two_part_barcode()
        ), "Feature Slice creation is only supported for slides with 2 part barcodes where bc1 \
            encodes x-axis and bc2 encodes y-axis"

        grid_index_of_barcode = compute_grid_index_of_barcode(matrix, slide)

        umi_slice = _compute_total_umis_slice(matrix, grid_index_of_barcode)

        metadata = HdFeatureSliceMetadata.new(
            sample_id, sample_desc, slide, transform_matrices_json
        )

        with h5.File(h5_path, "w") as f:
            f.attrs[h5_constants.H5_FILETYPE_KEY] = FEATURE_SLICE_H5_FILETYPE
            f.attrs[VERSION_ATTR_NAME] = FEATURE_SLICE_H5_VERSION
            # Save the metadata
            HdFeatureSliceIo._set_metadata(f, metadata)

            # Save the feature reference
            matrix.feature_ref.to_hdf5(f.create_group(h5_constants.H5_FEATURE_REF_ATTR))

            umi_group = f.create_group(UMIS_GROUP_NAME)
            # Total umi counts
            umi_slice.to_hdf5(
                umi_group.create_group(TOTAL_UMIS_GROUP_NAME),
                HdFeatureSliceMatrixMetadata.default(),
            )

            # Feature slices
            feature_slice_group = f.create_group(FEATURE_SLICES_GROUP_NAME)
            for feature_idx, feat_matrix in enumerate(
                HdFeatureSliceIo._create_feature_slices(matrix, grid_index_of_barcode)
            ):
                if len(feat_matrix.row) != 0:
                    feature_group = feature_slice_group.create_group(str(feature_idx))
                    feat_matrix.to_hdf5(
                        feature_group,
                        HdFeatureSliceMatrixMetadata.default(),
                    )

            # Total and corrected reads per barcode from the barcode summary h5
            if barcode_summary_h5 is not None:
                read_group = f.create_group(READS_GROUP_NAME)
                HdFeatureSliceIo._write_read_group(
                    read_group, grid_index_of_barcode, barcode_summary_h5
                )

    def set_transform_matrices(self, tranform_matrices: TransformMatrices):
        self.metadata.transform_matrices = tranform_matrices
        HdFeatureSliceIo._set_metadata(self.h5_file, self.metadata)

    def _get_or_create_group(self, group_name, group=None):
        group = group if group is not None else self.h5_file
        return group[group_name] if group_name in group else group.create_group(group_name)

    def read_filtered_mask(self, binning_scale: int = 1):
        """Read the mask of filtered barcodes."""
        assert self.open_mode == "r"
        return (
            self.load_counts_from_group_name(
                f"{MASKS_GROUP_NAME}/{FILTERED_SPOTS_GROUP_NAME}", binning_scale
            )
            > 0
        )

    def write_filtered_mask(
        self, slide: VisiumHdSlideWrapper, filtered_matrix: cr_matrix.CountMatrix
    ):
        """Write the mask of filtered barcodes."""
        assert self.open_mode == "a"
        grid_index_of_barcode = compute_grid_index_of_barcode(filtered_matrix, slide)

        self._write_filtered_mask(grid_index_of_barcode, FILTERED_SPOTS_GROUP_NAME, 1)

    def _write_filtered_mask(self, grid_index_of_barcode, group_name: str, binning_scale: int):
        filtered_mask = CooMatrix()

        for grid_index in grid_index_of_barcode:
            filtered_mask.insert(grid_index, 1)

        masks_group = self._get_or_create_group(MASKS_GROUP_NAME)
        filtered_mask.to_hdf5(
            masks_group.create_group(group_name),
            HdFeatureSliceMatrixMetadata.new(
                kind=MatrixDataKind.BINARY_MASK,
                domain=MatrixDataDomain.FILTERED_SPOTS,
                binning_scale=binning_scale,
            ),
        )

    def write_microscope_image_on_spots(self, image_path: str | None):
        self._write_image_on_spots(image_path, MICROSCOPE_IMAGE_GROUP_NAME)

    def write_cytassist_image_on_spots(self, image_path: str | None):
        self._write_image_on_spots(image_path, CYTASSIST_IMAGE_GROUP_NAME)

    def _write_image_on_spots(self, image_path: str | None, group_name):
        """Write the image to the feature slice h5."""
        img = image_util.cv_read_image_standard(image_path)

        # Use the CooMatrix for simplicity. Could be updated to store a 2d matrix later
        # Due to compression the difference is sizes would be small anyway
        image_matrix = CooMatrix()

        for (row, col), val in np.ndenumerate(img):
            image_matrix.insert(GridIndex2D(row=row, col=col), val)

        image_matrix.to_hdf5(
            self._get_or_create_group(IMAGES_GROUP_NAME).create_group(group_name),
            HdFeatureSliceMatrixMetadata.new(
                kind=MatrixDataKind.GRAY_IMAGE_U8,
                domain=MatrixDataDomain.ALL_SPOTS,
                binning_scale=1,
            ),
        )

    def write_binned_filtered_mask(self, bin_name: str, filtered_unbinned_bcs: list[bytes]):
        """Add binned filtered mask to the feature slice H5."""
        bin_size_um = int(bin_name.removeprefix("square_").removesuffix("um"))
        binning_scale = int(bin_size_um / self.metadata.spot_pitch)
        barcode_set = set()
        for bc in filtered_unbinned_bcs:
            barcode = SquareBinIndex(barcode=bc.decode())
            barcode_set.add((barcode.row // binning_scale, barcode.col // binning_scale))

        grid_index_of_barcode = [
            GridIndex2D(row=barcode[0], col=barcode[1]) for barcode in barcode_set
        ]
        self._write_filtered_mask(grid_index_of_barcode, bin_name, binning_scale)

    def write_secondary_analysis(self, bin_name: str, analysis_h5: str):
        """Add secondary analysis results at this bin level into the feature slice H5."""
        bin_size_um = bin_size_um_from_bin_name(bin_name)
        binning_scale = int(bin_size_um / self.metadata.spot_pitch)

        grid_index_of_barcode = []
        with h5.File(analysis_h5, "r") as f:
            for barcode in f["matrix"]["barcodes"][:]:
                barcode = SquareBinIndex(barcode=barcode.decode())
                grid_index_of_barcode.append(GridIndex2D(row=barcode.row, col=barcode.col))

            secondary_analysis_group = self._get_or_create_group("secondary_analysis")

            for group_name, dataset_name in [
                ("clustering", "clusters"),
                ("pca", "transformed_pca_matrix"),
                ("tsne", "transformed_tsne_matrix"),
                ("umap", "transformed_umap_matrix"),
            ]:
                if group_name in f:
                    analysis_h5_group = f[group_name]
                    slice_h5_group = self._get_or_create_group(group_name, secondary_analysis_group)

                    for subgroup_name in analysis_h5_group.keys():
                        analysis_data = analysis_h5_group[subgroup_name][dataset_name][()]
                        slice_h5_subgroup = slice_h5_group.create_group(
                            f"{bin_name}{subgroup_name}"
                        )

                        if analysis_data.ndim == 1:
                            self._write_analysis_data(
                                slice_h5_subgroup,
                                binning_scale,
                                grid_index_of_barcode,
                                analysis_data,
                                MatrixDataKind.CATEGORICAL,
                            )
                        elif analysis_data.ndim == 2:
                            for col in range(analysis_data.shape[1]):
                                self._write_analysis_data(
                                    slice_h5_subgroup.create_group(f"{col}"),
                                    binning_scale,
                                    grid_index_of_barcode,
                                    analysis_data[:, col],
                                    MatrixDataKind.FLOAT,
                                )

    def _write_analysis_data(
        self,
        group,
        binning_scale,
        grid_index_of_barcode,
        analysis_data,
        kind,
    ):
        assert len(analysis_data) == len(grid_index_of_barcode)
        analysis_data_matrix = CooMatrix()
        for grid_index, cluster in zip(grid_index_of_barcode, analysis_data):
            analysis_data_matrix.insert(grid_index, cluster)
        analysis_data_matrix.to_hdf5(
            group,
            HdFeatureSliceMatrixMetadata.new(
                kind=kind,
                domain=MatrixDataDomain.FILTERED_SPOTS,
                binning_scale=binning_scale,
            ),
        )

    def write_cas_analysis(self, bin_name, cas_csv_path):
        """Given a bin name and path to a cell annotation CSV, write it into the feature-slice matrix."""
        bin_size_um = bin_size_um_from_bin_name(bin_name)
        binning_scale = int(bin_size_um / self.metadata.spot_pitch)

        with open(cas_csv_path) as f:
            reader = csv.DictReader(f)
            sorted_cell_types = sorted(set(row[cas.COARSE_CELL_TYPES_KEY] for row in reader))

        cell_types_with_none_appended = [OUT_OF_TISSUE_SENTINEL] + sorted_cell_types
        cell_types_to_cell_type_idx = {y: x for x, y in enumerate(cell_types_with_none_appended)}
        grp_bin = self._get_or_create_group(CELL_ANNOTATIONS_GROUP_NAME).create_group(bin_name)
        grp_bin.create_dataset(
            CELL_TYPE_IDX_DATASET_NAME,
            data=np.array(cell_types_with_none_appended, dtype="S"),
            chunks=(cr_matrix.HDF5_CHUNK_SIZE,),
            maxshape=(None,),
            compression=cr_matrix.HDF5_COMPRESSION,
            shuffle=True,
        )

        cas_matrix = CooMatrix()
        with open(cas_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                bc = SquareBinIndex(row[cas.BARCODE_KEY])
                cas_matrix.insert(
                    GridIndex2D(row=bc.row, col=bc.col),
                    cell_types_to_cell_type_idx[row[cas.COARSE_CELL_TYPES_KEY]],
                )
        cas_matrix.to_hdf5(
            grp_bin.create_group(CELL_ANNOTATIONS_MATRIX_NAME),
            HdFeatureSliceMatrixMetadata.new(
                kind=MatrixDataKind.CATEGORICAL,
                domain=MatrixDataDomain.FILTERED_SPOTS,
                binning_scale=binning_scale,
            ),
        )

    def get_bins_with_cell_annotations(self):
        """Get bins with cell annotations."""
        if CELL_ANNOTATIONS_GROUP_NAME not in self.h5_file.keys():
            return []
        else:
            return [
                x
                for x in self.h5_file[CELL_ANNOTATIONS_GROUP_NAME].keys()
                if isinstance(self.h5_file[CELL_ANNOTATIONS_GROUP_NAME][x], h5.Group)
            ]

    def get_cell_annotations(self, bin_name):
        """Read cell annotation for a given bin name."""
        bin_size_um = bin_size_um_from_bin_name(bin_name)
        binning_scale = int(bin_size_um / self.metadata.spot_pitch)
        ncols_binned = int(np.ceil(self.ncols() / binning_scale))
        nrows_binned = int(np.ceil(self.nrows() / binning_scale))
        cell_annotation_matrix = CooMatrix.from_hdf5(
            self.h5_file[CELL_ANNOTATIONS_GROUP_NAME][bin_name][CELL_ANNOTATIONS_MATRIX_NAME]
        ).to_ndarray(nrows=nrows_binned, ncols=ncols_binned)
        cell_annotation_idx = [
            x.decode()
            for x in self.h5_file[CELL_ANNOTATIONS_GROUP_NAME][bin_name][
                CELL_TYPE_IDX_DATASET_NAME
            ][()]
        ]
        return (cell_annotation_idx, cell_annotation_matrix)
