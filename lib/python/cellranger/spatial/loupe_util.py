#!/usr/bin/env python
#
# Copyright (c) 2021 10X Genomics, Inc. All rights reserved.
#
"""Class and function to process Loupe manual alignment file."""

from __future__ import annotations

import base64
import itertools
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from cellranger.spatial.slide_design_o3 import (  # pylint: disable=no-name-in-module, import-error
    VisiumHdLayout,
    VisiumHdSlideWrapper,
)
from cellranger.spatial.tiffer import call_tiffer_checksum
from cellranger.spatial.transform import (
    normalize_perspective_transform,
    scale_from_transform_matrix,
    scale_matrix,
    transform_pts_2d,
    translation_matrix,
)


def get_remove_image_pages(loupe_alignment_file: str | None) -> set:
    """Set of image pages which should be skipped during tiling."""
    if loupe_alignment_file is None:
        return set()

    return set(LoupeParser(json_path=loupe_alignment_file).get_remove_image_pages())


def get_image_page_names(loupe_alignment_file: str | None) -> list:
    """Return channel names from alignment json."""
    if loupe_alignment_file is None:
        return []
    return LoupeParser(json_path=loupe_alignment_file).get_image_page_names()


def decode_spots_under_tissue(encoded_str: str):
    """Return an iterator of whether a spot in under tissue."""
    bitfield_bytes = base64.b64decode(encoded_str)
    return np.unpackbits(bytearray(bitfield_bytes)).astype(bool)


def encode_spots_under_tissue(list_of_in_tissue_bool_for_all_spots: Iterable[bool]) -> str:
    """Encode a list of bools into an encoded string."""
    in_tissue_grouped_in_eights = itertools.zip_longest(
        *([iter(list_of_in_tissue_bool_for_all_spots)] * 8), fillvalue=False
    )
    # Converts a group of 8 bits to uint8
    # for three bit (a hypothetical uint3) say [True, False, False]
    # shifts the first bit by 2 places to get b100 with the others being b000
    # summing up these numbers gives us uint3 representation of 4 as expected.
    bitfield_bytes = bytearray(
        sum(int(bit) << position for (position, bit) in enumerate(reversed(group_of_eight_bools)))
        for group_of_eight_bools in in_tissue_grouped_in_eights
    )
    return base64.b64encode(bitfield_bytes).decode()


class RequiredSpotDataDict(TypedDict):
    """Dict for required data from golang version of gpr spot data.

    .. code-block::golang

        type Spot struct {
            X            float32 `json:"x"`
            Y            float32 `json:"y"`
            Row          int     `json:"row"`
            Col          int     `json:"col"`
            Diameter     float32 `json:"dia"`
            Flag         int     `json:"flag,omitempty"`
            FiducialName string  `json:"fidName,omitempty"`
            ImageX       float32 `json:"imageX,omitempty"`
            ImageY       float32 `json:"imageY,omitempty"`
            Tissue       bool    `json:"tissue,omitempty"`
        }
    """

    x: float
    y: float
    dia: float
    row: int | None
    col: int | None


# total=False allows us to omit any keys without errors
class SpotDataDict(RequiredSpotDataDict, total=False):
    """Dict for data from golang version of gpr spot data.

    .. code-block::golang

        type Spot struct {
            X            float32 `json:"x"`
            Y            float32 `json:"y"`
            Row          int     `json:"row"`
            Col          int     `json:"col"`
            Diameter     float32 `json:"dia"`
            Flag         int     `json:"flag,omitempty"`
            FiducialName string  `json:"fidName,omitempty"`
            ImageX       float32 `json:"imageX,omitempty"`
            ImageY       float32 `json:"imageY,omitempty"`
            Tissue       bool    `json:"tissue,omitempty"`
        }
    """

    flag: int
    fidName: str
    imageX: float
    imageY: float
    tissue: bool
    name: int


class CytAssistInfoDict(TypedDict):
    checksumHiRes: str
    transformImages: list[list[float]]


class BaseAlignData(TypedDict):
    fiducial: list[SpotDataDict]
    oligo: list[SpotDataDict] | str


class SpotMetadataDict(TypedDict):
    """Metadata about how the "oligo" key is encoded."""

    bitfield_ordering: str
    bitfield_bit_convention: str
    bitfield_encoding: str
    bin_level: int  # in um


class HdSlideInfoDict(TypedDict):
    """Information used to reconstruct the VisiumHdSlideWrapper."""

    slide_name: str
    layout_str: str | None


# total=False allows us to omit any keys without errors
class AlignFileData(BaseAlignData, total=False):
    """Dict for required data from golang version of gpr spot data.

    .. code-block::golang

        type CommonExportData struct {
            Checksum         string         `json:"checksum"`
            CytAssistInfo    *CytAssistInfo `json:"cytAssistInfo,omitempty"`
            RemoveImagePages []int          `json:"removeImagePages"`
            ImagePageNames   []string       `json:"imagePageNames,omitempty"`
        }

        type AlignFileData struct {
            BaseAlignData
            CommonExportData
        }
    """

    transform: list[list[float]]
    serialNumber: str
    area: str
    checksum: str
    cytAssistInfo: CytAssistInfoDict
    removeImagePages: list[int]
    slide_layout_file: str
    spot_metadata: SpotMetadataDict
    hd_slide_info: HdSlideInfoDict


SLIDE_LAYOUT_FILE_KEY = "slide_layout_file"


@dataclass
class HdLayoutOffset:
    """Layout offset for HD in um."""

    x_offset: float
    y_offset: float

    def total_offset(self):
        """Return the norm of the offset."""
        return np.linalg.norm([self.x_offset, self.y_offset])


# pylint: disable=too-many-public-methods
class LoupeParser:
    """The class that can process Loupe json and json with similar format.

    The Loupe Json file has the form

    .. code-block::golang

        type AlignFileData struct {
            fiducial        gpr.SpotArray  `json:"fiducial"`
            oligo           gpr.SpotArray  `json:"oligo"`
            transform       [][]float64    `json:"transform"` // spots transform
            serialNumber    string         `json:"serialNumber"`
            area            string         `json:"area"`
            checksum        string         `json:"checksum"`
            cytAssistInfo   *CytAssistInfo `json:"cytAssistInfo"`
        }

        type CytAssistInfo struct {
            checksumHiRes   string      `json:"checksumHiRes"` // checked with tiffer
            transformImages [][]float64 `json:"transformImages"` // 3x3 transform matrix
        }

    The structure of the spot data is

    .. code-block::python

        {
            "fiducial": [
                {
                    "x": ,
                    "y": ,
                    "row": ,
                    "col": ,
                    "dia": ,
                    "imageX": ,
                    "imageY": ,
                    "fidName": ,
                },
                ...
            ],
            "oligo": [
                {
                    "x": ,
                    "y": ,
                    "row": ,
                    "col": ,
                    "dia": ,
                    "imageX": ,
                    "imageY": ,
                    "fidName": ,
                },
                ...
            ],
        }

    Attributes:
        _data_dict (dict): contain the original dictionary in the json
        _json_path (str): path of the loupe json file
    """

    def __init__(self, json_path: str | None = None, slide: VisiumHdSlideWrapper | None = None):
        """Initialize the parser.

        Args:
            json_path (str): path of the json file. Defaults to None.
            slide (VisiumHdSlideWrapper): HD slide design
        """
        has_json = json_path is not None
        has_slide = slide is not None
        assert has_json ^ has_slide, (
            "Exactly one of json_path or slide needs to be supplied to initialize LoupeParser class. "
            f"json_path supplied = {has_json}, slide supplied = {has_slide}"
        )

        self.hd_slide = None
        if has_json:
            with open(json_path) as f:
                self._data_dict = AlignFileData(**json.load(f))

            if SLIDE_LAYOUT_FILE_KEY in self._data_dict:
                vlf_str = base64.b64decode(self._data_dict[SLIDE_LAYOUT_FILE_KEY]).decode()
                area = (
                    self._data_dict["area"]
                    if "A1" in vlf_str or "B1" in vlf_str
                    else self._data_dict["area"][0]
                )
                hd_layout = VisiumHdLayout.from_vlf_str_and_area(vlf_str, area)

                if hd_layout.slide_design == "ct042723":
                    slide_name = "visium_hd_rc1"
                elif hd_layout.slide_design == "xl112023":
                    slide_name = "visium_hd_rcxl1"
                else:
                    slide_name = hd_layout.slide_design

                self.hd_slide = VisiumHdSlideWrapper(slide_name, hd_layout)

            elif "hd_slide_info" in self._data_dict:
                slide_info = self._data_dict["hd_slide_info"]
                self.hd_slide = VisiumHdSlideWrapper(
                    slide_info["slide_name"],
                    (
                        VisiumHdLayout.from_json_str(slide_info["layout_str"])
                        if slide_info["layout_str"]
                        else None
                    ),
                )
            elif self._data_dict.get("metadata", {}).get("slide_design") is not None:
                self.hd_slide = VisiumHdSlideWrapper(
                    self._data_dict.get("metadata", {}).get("slide_design"),
                    None,
                )

        elif has_slide:
            self.hd_slide = slide
            self._data_dict = AlignFileData(
                fiducial=[],
                oligo="",
                hd_slide_info=HdSlideInfoDict(
                    slide_name=slide.name(), layout_str=slide.layout_str()
                ),
            )

        self.loupe_alignment_json_version = None
        if loupe_version := self._data_dict.get("metadata", {}).get("version"):
            self.loupe_alignment_json_version = loupe_version

        # Checking that if the oligos are a string then we have a HD slide
        assert not (isinstance(self._data_dict.get("oligo"), str) ^ (self.hd_slide is not None)), (
            "Failure while building LoupeParser. We find inconsistencies between HD slide and metadata. "
            f"Has HD slide: {(self.hd_slide is not None)}, "
            f"Has HD oligo metadata {isinstance(self._data_dict.get('oligo'), str)}"
        )

    @staticmethod
    def from_visium_hd_slide(slide: VisiumHdSlideWrapper):
        """Construct the object from the Visium HD slide design."""
        return LoupeParser(slide=slide)

    def get_remove_image_pages(self) -> list:
        return self._data_dict.get("removeImagePages", [])

    def get_loupe_alignment_json_version(self) -> str | None:
        return self.loupe_alignment_json_version

    def get_image_page_names(self) -> list:
        return self._data_dict.get("imagePageNames", [])

    def has_area_id(self) -> bool:
        return "area" in self._data_dict

    def get_area_id(self) -> str | None:
        """Get the capture area ID."""
        assert self.has_area_id()
        area_id = self._data_dict["area"]
        if not area_id:  # make sure empty strings are interpreted as None
            area_id = None
        return area_id

    def set_area(self, area: str):
        """Sets capture area ID."""
        self._data_dict["area"] = area

    def hd_slide_layout(self) -> VisiumHdLayout | None:
        """Get the HD slide layout."""
        return self.hd_slide.layout() if self.hd_slide is not None else None

    def has_hd_slide(self) -> bool:
        """Returns if loupe util has a HD slide."""
        return self.hd_slide is not None

    def has_no_hd_layout(self) -> bool:
        """Returns true if the file had no slide info."""
        return (
            self._data_dict.get(SLIDE_LAYOUT_FILE_KEY) is None
            and self._data_dict.get("hd_slide_info", {}).get("layout_str") is None
        )

    def is_hd_unknown_slide(self) -> bool:
        """Returns if was a HD slide generated with a loupe JSON with unknown slide ID."""
        return self.has_hd_slide() and self.has_no_hd_layout()

    def has_serial_number(self) -> bool:
        return "serialNumber" in self._data_dict

    def get_serial_number(self) -> str | None:
        """Get the slide serial number."""
        assert self.has_serial_number()
        serial_number = self._data_dict["serialNumber"]
        if not serial_number:  # make sure empty strings are interpreted as None
            serial_number = None
        return serial_number

    def set_serial_number(self, serial_number: str):
        """Sets the slide serial number."""
        self._data_dict["serialNumber"] = serial_number

    def contain_fiducial_info(self) -> bool:
        """Whether the data contain the fiducial information.

        Returns:
            bool: whether contain the entire fiducial information.
        """
        return ("fiducial" in self._data_dict) and (len(self._data_dict["fiducial"]) > 0)

    def regist_target_checksum(self) -> str | None:
        return self._data_dict.get("cytAssistInfo", {}).get("checksumHiRes", None)

    def fiducial_image_checksum(self) -> str | None:
        return self._data_dict.get("checksum", None)

    def set_checksum(self, checksum):
        self._data_dict["checksum"] = checksum

    def contain_spots_info(self) -> bool:
        """Whether the data contain the spots information.

        The current behavior of loupe is that when no manual fiducial registration and tissue
        selection is performed and only manual tissue registration is done, the key `fiducial`
        and `oligo` still exist but the value will be an empty list. Note this is a different
        behavior with when no manual tissue registration is done.

        Returns:
            bool: whether contain the entire spots information.
        """
        return len(self._data_dict["oligo"]) > 0

    def contain_cyta_info(self) -> bool:
        """Return the CytaAssist information.

        The current behavior of loupe is that when no manual tissue registration is done,
        The key `cytAssistInfo` won't appear in the json.

        Returns:
            bool: whether the loupe json contains tissue registration information.
        """
        return "cytAssistInfo" in self._data_dict

    def oligos_preselected(self) -> bool:
        """Whether any tissue oligos are already selected.

        Returns:
            bool: True if tissue oligos are selected
        """
        assert "oligo" in self._data_dict
        if not self.contain_spots_info():
            return False
        oligo = self._data_dict["oligo"]
        if isinstance(oligo, str):
            return any(decode_spots_under_tissue(oligo))
        else:
            for pt_dict in oligo:
                selected = pt_dict.get("tissue", False)
                if selected:
                    return True
        return False

    def _encode_spots_under_tissue(self, spots_under_tissue: Iterable[bool]):
        """Encode a list of bools into an encoded string."""
        self._data_dict["oligo"] = encode_spots_under_tissue(spots_under_tissue)
        self._data_dict["spot_metadata"] = SpotMetadataDict(
            bitfield_ordering="row-major",
            bitfield_bit_convention="top-left",
            bitfield_encoding="base64",
            bin_level=2,  # in um
        )
        self._data_dict["spot_count"] = self.hd_slide.num_spots()

    def set_all_tissue_oligos(self, under_tissue: bool):
        """For each oligo spot, set tissue to True adding the key if needed."""
        assert "oligo" in self._data_dict
        if isinstance(self._data_dict["oligo"], list) and self.contain_spots_info():
            for pt_dict in self._data_dict["oligo"]:
                pt_dict["tissue"] = under_tissue
        elif isinstance(self._data_dict["oligo"], str):
            assert self.hd_slide is not None
            self._encode_spots_under_tissue(under_tissue for _ in range(self.hd_slide.num_spots()))
        else:
            raise ValueError(
                "Invalid Loupe File!! Oligos found are not a list (expected for SD) or a string (expected for HD.)"
            )

    def get_spots_data(self, scale: float = 1) -> dict[str, list[SpotDataDict]]:
        """Return the data containing both fiducial and oligo data.

        Args:
            scale (float): the scale factor to apply to the positions of
                all the spots.

        Returns:
            Dict: dictionary that contains the spot data
        """
        output_dict = {}

        if self.contain_fiducial_info():
            output_dict["fiducial"] = self._data_dict["fiducial"].copy()
            for pt_dict in output_dict["fiducial"]:
                if "imageX" in pt_dict:
                    assert "imageY" in pt_dict
                    pt_dict["imageX"] = pt_dict["imageX"] * scale
                    pt_dict["imageY"] = pt_dict["imageY"] * scale
                pt_dict["dia"] = pt_dict["dia"] * scale

        if isinstance(self._data_dict["oligo"], str):
            output_dict = self._data_dict.copy()
            transform = normalize_perspective_transform(
                scale_matrix(scale) @ np.array(output_dict["transform"])
            )
            output_dict["transform"] = transform.tolist()
        else:
            output_dict["oligo"] = self._data_dict["oligo"].copy()
            for pt_dict in output_dict["oligo"]:
                if "imageX" in pt_dict:
                    assert "imageY" in pt_dict
                    pt_dict["imageX"] = pt_dict["imageX"] * scale
                    pt_dict["imageY"] = pt_dict["imageY"] * scale
                pt_dict["dia"] = pt_dict["dia"] * scale
        return output_dict

    def get_cyta_data(self, scale: float = 1) -> dict:
        """Get the registration target image to cytassist image transform.

        Args:
            scale (float, optional): Inverse of regist_target_scalef. Defaults to 1.

        Returns:
            dict: CytaAssist info contained in Loupe
        """
        assert "cytAssistInfo" in self._data_dict
        output_dict = {"cytAssistInfo": self._data_dict["cytAssistInfo"]}
        if scale != 1:
            # fullres tissue -> cytassist transform
            original_transform_mat = output_dict["cytAssistInfo"]["transformImages"]
            original_transform_mat = np.array(original_transform_mat)
            # registration target -> cytassist
            scaled_transform = normalize_perspective_transform(
                original_transform_mat @ scale_matrix(scale)
            )
            output_dict["cytAssistInfo"]["transformImages"] = scaled_transform.tolist()
        return output_dict

    def get_cyta_transform(self) -> list[list[float]]:
        """Get the image transform of cytassist."""
        assert "cytAssistInfo" in self._data_dict
        return self._data_dict["cytAssistInfo"]["transformImages"]

    def get_fiducials_data(self) -> list[SpotDataDict]:
        """Get the fiducial data."""
        if self.hd_slide is None:
            if "fiducial" not in self._data_dict:
                raise KeyError("Data doesn't contain fiduical information")
            return self._data_dict["fiducial"]
        else:
            return [
                SpotDataDict(
                    x=f.center.x,
                    y=f.center.y,
                    dia=self.hd_slide.circular_fiducial_outer_radius() * 2,
                    row=None,
                    col=None,
                    name=f.code,
                )
                for f in self.hd_slide.circular_fiducials()
            ]

    def get_oligos_data(self) -> list[SpotDataDict]:
        """Get the oligo data."""
        if "oligo" not in self._data_dict:
            raise KeyError("Data doesn't contain oligo information")
        return self._data_dict["oligo"]

    def has_spot_transform(self) -> bool:
        return "transform" in self._data_dict

    def get_spot_transform(self) -> np.ndarray:
        """Get spot transform data."""
        if self.has_spot_transform():
            return np.array(self._data_dict["transform"])
        else:
            return np.eye(3)

    def set_spot_transform(self, transform: list[list[float]]):
        """Set spot transform."""
        self._data_dict["transform"] = transform

    def get_fiducials_xy(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Get the design x, y of fiducials."""
        fids_xy = [[pt_dict["x"], pt_dict["y"]] for pt_dict in self._data_dict["fiducial"]]
        return np.array(fids_xy)

    def get_oligos_xy(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Get the design x, y of the oligos."""
        oligos_xy = [[pt_dict["x"], pt_dict["y"]] for pt_dict in self._data_dict["oligo"]]
        return np.array(oligos_xy)

    def get_fiducials_rowcol(self) -> np.ndarray:
        """Get the row and col of fiducials."""
        if self.hd_slide is not None:
            return np.array([[i, 0] for i, _ in enumerate(self.hd_slide.circular_fiducials())])
        fids_rc = [[pt_dict["row"], pt_dict["col"]] for pt_dict in self._data_dict["fiducial"]]
        return np.array(fids_rc)

    def get_oligos_rowcol(self) -> np.ndarray:
        """Get the row and col of oligos."""
        if self.hd_slide is not None:
            return np.array(
                [
                    [
                        spot.grid_index.row,
                        spot.grid_index.col,
                    ]
                    for spot in self.hd_slide.spots()
                ]
            )
        else:
            return np.array(
                [[pt_dict["row"], pt_dict["col"]] for pt_dict in self._data_dict["oligo"]]
            )

    def get_fiducials_imgxy(
        self, scale: float = 1
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Get the position of fiducials in the image."""
        if self.hd_slide is not None:
            transform_mat = normalize_perspective_transform(
                scale_matrix(scale) @ self.get_spot_transform()
            )
            fids_xy = np.array(
                [[fid.center.x, fid.center.y] for fid in self.hd_slide.circular_fiducials()]
            )
            fids_xy = transform_pts_2d(fids_xy, transform_mat)
        else:
            fids_xy = [
                [pt_dict["imageX"], pt_dict["imageY"]] for pt_dict in self._data_dict["fiducial"]
            ]
            fids_xy = np.array(fids_xy)
            fids_xy = fids_xy * scale
        return fids_xy

    def get_oligos_imgxy(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Get the position of oligos in the image."""
        if self.hd_slide is not None:
            return self.hd_slide.spot_xy_with_transform(self.get_spot_transform())
        else:
            return np.array(
                [[pt_dict["imageX"], pt_dict["imageY"]] for pt_dict in self._data_dict["oligo"]]
            )

    def get_fiducials_diameter(self, scale: float = 1) -> float:
        """Get the diameter of the fiducials.

        Returns 0 if there are no fiducials.
        """
        if self.hd_slide is not None:
            radius = self.hd_slide.circular_fiducial_outer_radius()
            radius = radius if radius is not None else 0.0
            return radius * 2 * scale_from_transform_matrix(self.get_spot_transform()) * scale
        return self._data_dict["fiducial"][0]["dia"] * scale

    def get_oligos_diameter(self) -> float:
        """Get the diameter of the oligos."""
        if self.hd_slide is not None:
            return self.hd_slide.spot_size() * scale_from_transform_matrix(
                self.get_spot_transform()
            )
        return self._data_dict["oligo"][0]["dia"]

    def _hd_tissue_oligos_flags(self) -> list[bool]:
        # Sanity checks
        assert self.hd_slide is not None
        metadata = self._data_dict["spot_metadata"]
        assert metadata["bitfield_ordering"] == "row-major"
        assert metadata["bitfield_bit_convention"] == "top-left"
        assert metadata["bitfield_encoding"] == "base64"

        bin_scale = int(metadata["bin_level"] / self.hd_slide.spot_pitch())
        ncols_binned = self.hd_slide.num_cols(bin_scale)
        nrows_binned = self.hd_slide.num_rows(bin_scale)
        spot_count = self._data_dict["spot_count"]
        assert spot_count == (nrows_binned * ncols_binned)

        in_tissue_flag = np.reshape(
            decode_spots_under_tissue(self._data_dict["oligo"])[:spot_count],
            (nrows_binned, ncols_binned),
        )

        return [
            in_tissue_flag[spot.grid_index.row // bin_scale, spot.grid_index.col // bin_scale]
            for spot in self.hd_slide.spots()
        ]

    def tissue_oligos_flags(self) -> list[bool]:
        """Return whether each oligo is under the tissue."""
        if self.hd_slide is not None:
            return self._hd_tissue_oligos_flags()

        return [pt_dict.get("tissue", False) for pt_dict in self._data_dict["oligo"]]

    def update_fiducials_imgxy(self, imgxy: np.ndarray) -> None:
        """Set the x,y position of each fiducial in the image."""
        num_fid = len(self._data_dict["fiducial"])
        if num_fid != len(imgxy):
            raise RuntimeError(
                f"Number of fiducials {num_fid} in data and {len(imgxy)} in coordinates don't match"
            )
        for pt_dict, (x, y) in zip(self._data_dict["fiducial"], imgxy):
            pt_dict["imageX"] = x
            pt_dict["imageY"] = y

    def update_oligos_imgxy(self, imgxy: np.ndarray) -> None:
        """Set the x,y position of each oligo in the image."""
        num_oligos = len(self._data_dict["oligo"])
        if num_oligos != len(imgxy):
            raise RuntimeError(
                f"Number of oligos {num_oligos} in data and {len(imgxy)} in coordinates don't match"
            )
        for pt_dict, (x, y) in zip(self._data_dict["oligo"], imgxy):
            pt_dict["imageX"] = x
            pt_dict["imageY"] = y

    def update_tissue_oligos(self, tissue_oligos_flag: list[bool]) -> None:
        """Set whether each oligo is under the tissue."""
        if self.hd_slide is None:
            num_oligos = len(self._data_dict["oligo"])
            if num_oligos != len(tissue_oligos_flag):
                raise RuntimeError(
                    f"Number of oligos {num_oligos} in data and {len(tissue_oligos_flag)} in flags don't match"
                )
            for pt_dict, flag in zip(self._data_dict["oligo"], tissue_oligos_flag):
                pt_dict["tissue"] = bool(flag)
        else:
            self._encode_spots_under_tissue(tissue_oligos_flag)

    def update_spots_dia_by_scale(self, scale: float):
        """Update the diameter of all spots based on the scale."""
        for pt_dict in self._data_dict["fiducial"]:
            pt_dict["dia"] = pt_dict["dia"] * scale
        for pt_dict in self._data_dict["oligo"]:
            pt_dict["dia"] = pt_dict["dia"] * scale

    def update_tissue_transform(self, transform: list[list[float]], tissue_image_path: str):
        """Adds Tissue Registration Transform and checksum if not already in json_file."""
        if "cytAssistInfo" not in self._data_dict:
            self._data_dict["cytAssistInfo"] = {
                "checksumHiRes": call_tiffer_checksum(tissue_image_path),
                "transformImages": transform,
            }

    def update_checksum(self, cytassist_image_path: str):
        """Adds Cytassist Image checksum to json_file if not already there."""
        if "checksum" not in self._data_dict:
            self._data_dict["checksum"] = call_tiffer_checksum(cytassist_image_path)

    def save_to_json(self, filepath: str):
        """Save the data to json file."""
        with open(filepath, "w") as f:
            json.dump(self._data_dict, f, indent=4)

    def transform(self, transform_mat: np.ndarray[tuple[int, int], np.dtype[np.float32]]):
        """Transform spot and fiducial data based on the transform_mat."""
        if self.hd_slide is None:
            scale = scale_from_transform_matrix(transform_mat)
            # Transform the design spots to the image space
            fid_xy = self.get_fiducials_xy()
            oligo_xy = self.get_oligos_xy()
            fid_imgxy = transform_pts_2d(fid_xy, transform_mat)
            oligo_imgxy = transform_pts_2d(oligo_xy, transform_mat)

            self.update_fiducials_imgxy(fid_imgxy)
            self.update_oligos_imgxy(oligo_imgxy)
            self.update_spots_dia_by_scale(scale)
        else:
            self._data_dict["transform"] = transform_mat.tolist()

    def update_hd_layout_with_offset(self, offset: HdLayoutOffset):
        """Update the HD layout with the offset. Caller needs to ensure that the layout is not None."""
        if self.hd_slide is None:
            raise RuntimeError("No HD slide found in LoupeParser.")
        layout = self.hd_slide.layout()
        if layout is None:
            raise RuntimeError("No HD layout found in LoupeParser.")

        layout.transform = layout.transform @ translation_matrix(offset.x_offset, offset.y_offset)
        self.hd_slide = VisiumHdSlideWrapper(
            self.hd_slide.name(),
            layout,
        )

    @staticmethod
    def estimate_mem_gb_from_json_file(json_file: str | None) -> float:
        """Estimate the memory required to create the LoupeParser instance from a json file.

        Args:
            json_file (str): Path to the json file
        Returns:
            estimated memory in GB. Returns 0 if the argument is none or the
            file does not exist.
        """
        if json_file is None or (not os.path.exists(json_file)):
            return 0.0
        # Empirically estimated memory by loading the json file and checking
        # the RSS used
        mem_gb_per_gb_on_disk = 4.0
        file_size_gb = os.path.getsize(json_file) / (1024**3)
        return mem_gb_per_gb_on_disk * file_size_gb
