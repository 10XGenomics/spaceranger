#
# Copyright (c) 2025 10X Genomics, Inc. All rights reserved.
#

"""Get differential expression JSON from diff exp CSV."""

import json
import os
from dataclasses import asdict, dataclass

import martian

from cellranger.analysis.analysis_types import (
    DifferentialExpression,
    DifferentialExpressionWithFeatures,
)
from cellranger.websummary.analysis_tab_core import diffexp_table
from cellranger.websummary.react_components import ReactComponentEncoder

__MRO__ = """
stage GET_SEGMENTED_DIFFEXP_JSON(
    in  path analysis_csv,
    out json diffexp,
    src py   "stages/spatial/get_segmented_diffexp_json",
)
"""


@dataclass
class DifferentialExpressionTable:
    """Differential expression table for a single bin size and clustering algorithm."""

    table: dict


def main(args, outs):
    if not args.analysis_csv or not os.path.exists(
        diffexp_csv_path := os.path.join(
            args.analysis_csv,
            "diffexp",
            "gene_expression_graphclust",
            "differential_expression.csv",
        )
    ):
        martian.clear(outs)
        return

    diffexp_with_features = DifferentialExpressionWithFeatures.from_diffexp_csv(diffexp_csv_path)
    num_clusters = DifferentialExpression.get_number_of_clusters_from_csv(diffexp_csv_path)

    cluster_names = [f"Cluster {i}" for i in range(1, num_clusters + 1)]

    diffexp_table_dict = DifferentialExpressionTable(
        table=json.loads(
            json.dumps(
                diffexp_table(
                    diffexp_with_features=diffexp_with_features,
                    cluster_names=cluster_names,
                ),
                cls=ReactComponentEncoder,
            )
        )
    )

    with open(outs.diffexp, "w") as f:
        json.dump(asdict(diffexp_table_dict), f)
