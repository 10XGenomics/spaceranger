#!/usr/bin/env python
#
# Copyright (c) 2022 10X Genomics, Inc. All rights reserved.
#
"""Utilities for use in spatial deconvolution."""

from __future__ import annotations

import csv
from typing import Protocol

import numpy as np


# pylint: disable=too-few-public-methods
class ScikitModel(Protocol):
    components_: np.ndarray


def get_lda_feature_counts(
    model: ScikitModel, feature_names: list, n_features: int, normalized: bool = False
) -> dict:
    """Creates a dictionary of features and LDA counts for each deconvolution topic.

    Args:
        model (model.fit): LDA model fit to GEX matrix
        feature_names (list): list of GEX feature names to assign as keys to LDA values
        n_features (int, optional): Number of features to return for each topic. Defaults to 100.
        normalized (bool, optional): Normalize the LDA values and multiples that number by 1 million.
        Normalizes to 1M feature observations per topic. Defaults to False.

    Returns:
        dict: for each topic a dict of tuple(features and values (LDA proportions))
    """
    if normalized:
        model_components = (model.components_ / model.components_.sum(axis=1)[:, np.newaxis]) * 1e6
    else:
        model_components = model.components_
    # Getting indices of all genes which are in top n_features of any topic
    # argsort(-model_components) to get argsort in descending order
    indices_to_keep = sorted(set(np.argsort(-model_components, axis=1)[:, :n_features].flatten()))
    topic_gene_distribution = {}
    feature_names = np.array(feature_names)
    for i, topic in enumerate(model_components):
        topic_gene_distribution[i] = list(
            zip(feature_names[indices_to_keep], topic[indices_to_keep])
        )
    return topic_gene_distribution


def read_topic_header(csv_file: str) -> list[str]:
    """Read the topic names from the header of a CSV file.

    Args:
        csv_file (str): The path to the CSV file.

    Returns:
        list[str]: A list of topic names extracted from the header.

    Raises:
        FileNotFoundError: If the specified CSV file is not found.

    """
    with open(csv_file) as file:
        reader = csv.reader(file)
        header_row = next(reader)  # Read the first row

    topic_columns = [col for col in header_row if col.startswith("Feature count topic")]

    # Extract the topic names
    topic_names = [col.replace("Feature count ", "") for col in topic_columns]

    return topic_names
