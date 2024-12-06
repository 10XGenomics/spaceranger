#
# Copyright (c) 2019 10X Genomics, Inc. All rights reserved.
#
"""Utilities designed to help create the web summary used in Spatial."""


from __future__ import annotations

import os

import cellranger.metrics_names as metrics_names
import cellranger.report as cr_report  # pylint: disable=no-name-in-module
import cellranger.rna.library as rna_library
import cellranger.webshim.common as cr_webshim
import cellranger.websummary.analysis_tab_core as cr_atc
import cellranger.websummary.sample_properties as sp
import cellranger.websummary.web_summary_builder as ws_builder
from cellranger.reference_paths import get_reference_genomes
from cellranger.spatial.data_utils import (
    IMAGEX_LOWRES,
    IMAGEY_LOWRES,
    get_lowres_coordinates,
    get_scalefactors,
)
from cellranger.spatial.image import WebImage
from cellranger.spatial.loupe_util import LoupeParser
from cellranger.targeted.targeted_constants import TARGETING_METHOD_HC, TARGETING_METHOD_TL
from cellranger.websummary.metrics import (
    SpatialHDTemplateLigationMetricAnnotations,
    SpatialMetricAnnotations,
    SpatialTargetedMetricAnnotations,
    SpatialTemplateLigationMetricAnnotations,
    output_metrics_csv_from_annotations,
)
from cellranger.websummary.spatial_utils import (
    SPATIAL_COMMAND_NAME,
    SPATIAL_PIPELINE_NAME,
    SpatialReportArgs,
    tissue_plots_by_clustering_spatial,
    umi_on_spatial_plot,
)

SENSITIVITY_PLOT_HELP = {
    "data": [
        [
            "",
            [
                "(left) Total UMI counts for each spot overlayed on the tissue image. "
                "Spots with greater UMI counts likely have higher RNA content than spots "
                "with fewer UMI counts. ",
                "(right) Total UMI counts for spots displayed by a 2-dimensional embedding produced "
                "by the t-SNE algorithm. In this space, pairs of spots that are close to each other "
                "have more similar gene expression profiles than spots that are distant from each "
                "other.",
            ],
        ]
    ],
    "title": "UMIs Detected",
}


def create_common_spatial_summaries(args, outs, metrics_csv_out=None):
    """This method generates the CS portion of the summarize stage outputs.

    it is shared by both the PD and CS code.
    """
    write_merged_metrics_json(args, outs)

    ref_genomes = get_reference_genomes(args.reference_path)
    redundant_loupe_alignment = _is_loupe_alignment_redundant(
        args.loupe_alignment_file, args.cytassist_image_paths, args.tissue_image_paths
    )
    sample_properties = sp.ExtendedCountSampleProperties(
        sample_id=args.sample_id,
        sample_desc=args.sample_desc,
        genomes=ref_genomes,
        reference_path=args.reference_path,
        chemistry_defs=args.chemistry_defs,
        is_spatial=True,
        target_set=args.target_set_name,
        target_panel_summary=args.target_panel_summary,
        feature_ref_path=args.feature_reference,
        reorientation_mode=args.reorientation_mode,
        loupe_alignment_file=args.loupe_alignment_file,
        filter_probes=args.filter_probes,
        aligner=args.aligner,
        include_introns=args.include_introns,
        redundant_loupe_alignment=redundant_loupe_alignment,
        v1_pattern_fix=args.v1_pattern_fix is not None,
        default_layout=args.hd_layout_data_json is None,
        override_id=args.override_id,
        slide_id_mismatch=args.slide_id_mismatch,
        is_visium_hd=args.is_visium_hd or False,
        cmdline=os.environ.get("CMDLINE", "NA"),
        cytassist_run_metrics=args.cytassist_run_metrics,
        itk_error_string=args.itk_error_string,
    )
    sample_data_paths = sp.SampleDataPaths(
        summary_path=outs.metrics_summary_json,
        barcode_summary_path=args.barcode_summary_h5,
        analysis_path=args.analysis,
        filtered_barcodes_path=args.filtered_barcodes,
        antibody_histograms_path=args.antibody_histograms,
        antibody_treemap_path=args.antibody_treemap,
        raw_normalized_heatmap_path=args.raw_normalized_heatmap,
        isotype_scatter_path=args.isotype_scatter,
        gex_fbc_correlation_heatmap_path=args.gex_fbc_correlation_heatmap,
        feature_metrics_path=args.per_feature_metrics_csv,
    )
    sample_data = cr_webshim.load_sample_data(sample_properties, sample_data_paths)

    spatial_args = SpatialReportArgs(
        sample_id=args.sample_id,
        sample_desc=args.sample_desc,
        tissue_lowres_image=args.tissue_lowres_image,
        scalefactors=args.scalefactors,
        matrix=args.matrix,
        tissue_positions=args.tissue_positions,
        detected_tissue_image=args.detected_tissue_image,
        qc_resampled_cyta_img=args.qc_resampled_cyta_img,
        qc_regist_target_img=args.qc_regist_target_img,
        analysis=args.analysis,
        target_set_name=args.target_set_name,
        target_panel_summary=args.target_panel_summary,
        targeting_method=args.targeting_method,
        feature_ref_path=args.feature_reference,
        reorientation_mode=args.reorientation_mode,
        loupe_alignment_file=args.loupe_alignment_file,
        filter_probes=args.filter_probes,
        aligner=args.aligner,
    )

    # Determine which spatial metrics csv should be used
    spatial_mets = (
        # Hyb capture spatial metrics
        # Also use hybcap targeted metrics for 3'GEX with a probe set and with STAR aligner,
        # which is used to support aggr'ing 3'GEX with RTL.
        SpatialTargetedMetricAnnotations()
        if args.targeting_method == TARGETING_METHOD_HC
        or (args.targeting_method == TARGETING_METHOD_TL and args.aligner == "star")
        # RTL spatial metrics
        else (
            SpatialTemplateLigationMetricAnnotations()
            if args.targeting_method == TARGETING_METHOD_TL
            and args.aligner != "star"
            and not sample_properties.is_visium_hd
            else (
                SpatialHDTemplateLigationMetricAnnotations()
                if args.targeting_method == TARGETING_METHOD_TL
                and args.aligner != "star"
                and sample_properties.is_visium_hd
                # Regular spatial metrics
                else SpatialMetricAnnotations()
            )
        )
    )

    web_sum_data = build_web_summary_data_spatial(sample_properties, sample_data, spatial_args)
    if metrics_csv_out:
        output_metrics_csv_from_annotations(
            spatial_mets, sample_data, metrics_csv_out, sample_properties.genomes
        )
    return web_sum_data


def _is_loupe_alignment_redundant(loupe_alignment_file, cytassist_image_paths, tissue_image_paths):
    if loupe_alignment_file is not None:
        loupe_data = LoupeParser(json_path=loupe_alignment_file)
        if (
            loupe_data.contain_cyta_info()
            and loupe_data.contain_fiducial_info()
            and any(cytassist_image_paths)
            and not any(tissue_image_paths)
        ):
            return True
    return False


def _add_image_alignment_alarms(web_sum_data, sample_data, metadata):
    """Adds a warning if issues detected with image registration.

    :param web_sum_data:
    :param sample_data:
    :param metadata: A SpatialMetricsAnnotation instance
    :return:
    """
    assert isinstance(metadata, SpatialMetricAnnotations)
    alarms = metadata.gen_metric_list(
        sample_data.summary,
        [metrics_names.SUSPECT_ALIGNMENT, metrics_names.REORIENTATION_NEEDED],
        [],
    )
    alarm_dicts = [metric.alarm_dict for metric in alarms if metric.alarm_dict]
    if alarm_dicts:
        web_sum_data.alarms.extend(alarm_dicts)


SD_WEBSUMMARY_IMAGE_WIDTH = 470
HD_WEBSUMMARY_IMAGE_WIDTH = 1500
WEBSUMMARY_REGISTRATION_QC_SLIDER_WIDTH = 470


def build_web_summary_data_spatial(sample_properties, sample_data, spatial_args):
    # pylint: disable=invalid-name,too-many-locals,missing-function-docstring
    if sample_properties.is_targeted:
        if (
            spatial_args.targeting_method == TARGETING_METHOD_TL
            and not sample_properties.is_visium_hd
        ):
            metadata = SpatialTemplateLigationMetricAnnotations()
        elif (
            spatial_args.targeting_method == TARGETING_METHOD_TL and sample_properties.is_visium_hd
        ):
            metadata = SpatialHDTemplateLigationMetricAnnotations()
        else:
            metadata = SpatialTargetedMetricAnnotations()
    else:
        metadata = SpatialMetricAnnotations(intron_mode_alerts=sample_properties.include_introns)

    new_width = HD_WEBSUMMARY_IMAGE_WIDTH if sample_data.is_visium_hd else SD_WEBSUMMARY_IMAGE_WIDTH

    detected_tissue_image = WebImage(spatial_args.detected_tissue_image)
    small_img = detected_tissue_image.resize_and_encode_image(new_width=new_width)
    zoom_images = {
        "small_image": small_img.base64_encoded_str,
        "big_image": detected_tissue_image.base64_encoded_str,
        "sizes": {"width": detected_tissue_image.width, "height": detected_tissue_image.height},
        "plot_title": "Tissue Detection and Fiducial Alignment",
    }

    if spatial_args.qc_resampled_cyta_img and spatial_args.qc_regist_target_img:
        qc_resampled_cyta_img = WebImage(
            spatial_args.qc_resampled_cyta_img
        ).resize_and_encode_image(new_width=new_width)
        qc_regist_target_img = WebImage(spatial_args.qc_regist_target_img).resize_and_encode_image(
            new_width=new_width
        )
        regist_images = {
            "imgA": qc_resampled_cyta_img.base64_encoded_str,
            "imgATitle": "CytAssist Image",
            "imgB": qc_regist_target_img.base64_encoded_str,
            "imgBTitle": "Microscope Image",
            "sizes": {"width": WEBSUMMARY_REGISTRATION_QC_SLIDER_WIDTH},
            "plot_title": "CytAssist Image Alignment",
            "slider_title": "",
        }
    else:
        regist_images = None

    web_sum_data = ws_builder.build_web_summary_data_common(
        sample_properties,
        sample_data,
        SPATIAL_PIPELINE_NAME,
        metadata,
        SPATIAL_COMMAND_NAME,
        zoom_images,
        regist_images,
    )

    _add_image_alignment_alarms(web_sum_data, sample_data, metadata)

    if not (spatial_args.tissue_positions and os.path.exists(spatial_args.tissue_positions)):
        return web_sum_data

    # Add in the clustering over tissue plots to the left pane of the clustering selector
    scalef = get_scalefactors(spatial_args.scalefactors)
    coords = get_lowres_coordinates(spatial_args.tissue_positions, spatial_args.scalefactors)
    l, r = coords[IMAGEX_LOWRES].min(), coords[IMAGEX_LOWRES].max()
    t, b = coords[IMAGEY_LOWRES].min(), coords[IMAGEY_LOWRES].max()
    hoffset = 0.13 * (r - l + 1)  # arbitrary looking numbers due to difference
    voffset = 0.10 * (b - t + 1)  # in vertical vs. horizontal packing

    # ensure that the crop box is within the image
    # (plotly does weird things otherwise)
    cb_x0 = max(l - hoffset, 0)
    cb_x1 = min(r + hoffset, detected_tissue_image.width - 1)
    cb_y0 = max(t - voffset, 0)
    cb_y1 = min(b + voffset, detected_tissue_image.height - 1)

    # compute marker size for this image
    plot_height = 265.5  # this is fixed for our design, but in a better world we'd
    # get this from Plotly.  If this works, we can consider
    # just updating markerref in javascript instead of doing this
    spot_diameter_image = scalef["tissue_lowres_scalef"] * scalef["spot_diameter_fullres"]
    scaled_spot_diameter = spot_diameter_image * plot_height / (cb_y1 - cb_y0)

    lowres_tissue = WebImage(
        spatial_args.tissue_lowres_image,
        cropbox=[cb_x0, cb_x1, cb_y0, cb_y1],
        markersize=scaled_spot_diameter,
    )

    if web_sum_data.clustering_selector:
        web_sum_data.clustering_selector.left_plots = tissue_plots_by_clustering_spatial(
            sample_data, coords, lowres_tissue
        )
    if web_sum_data.antibody_clustering_selector:
        web_sum_data.antibody_clustering_selector.left_plots = tissue_plots_by_clustering_spatial(
            sample_data, coords, lowres_tissue, library_type=rna_library.ANTIBODY_LIBRARY_TYPE
        )

    # Make UMI Count plots
    ## Spatial only uses GEX counts
    umi_spatial_plot = umi_on_spatial_plot(
        sample_data, coords, lowres_tissue, library_type=rna_library.GENE_EXPRESSION_LIBRARY_TYPE
    )
    umi_tsne_plot = cr_atc.umi_on_tsne_plot(sample_data, spatial=True)
    if umi_spatial_plot and umi_tsne_plot:
        web_sum_data.analysis_tab["umi_plots"] = {
            "help_txt": SENSITIVITY_PLOT_HELP,
            "umi_spatial_plot": umi_spatial_plot,
            cr_atc.UMI_TSNE_PLOT: umi_tsne_plot,
        }
    return web_sum_data


def write_merged_metrics_json(args, outs):
    # pylint: disable=missing-function-docstring
    # Both manual and automatic alignment pathways should generate the
    # fraction of spots under tissue metric
    assert args.fraction_under_tissue is not None
    alignment_dict = {
        "aligned_fiducials": args.aligned_fiducials,
        "fraction_under_tissue": args.fraction_under_tissue,
    }
    id_dict = {
        "sample_id": args.sample_id,
        "sample_desc": args.sample_desc,
        "spatial_slide_info": args.slide_serial_info,
    }
    reorientation_dict = {"reorientation_mode": args.reorientation_mode}
    loupe_alignment_dict = {"loupe_alignment_file": args.loupe_alignment_file}
    cr_report.merge_jsons(
        args.summaries,
        outs.metrics_summary_json,
        [alignment_dict, id_dict, reorientation_dict, loupe_alignment_dict],
    )
