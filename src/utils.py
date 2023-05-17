"""Utils."""
from __future__ import annotations

import datetime
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp


def read_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads data from csv files located at given path.

    Parameters
    ----------
    path : str
        The path where the csv files are located.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing three pandas DataFrames representing the data, users,
        and items csv files, respectively.

    Raises
    ------
    AssertionError
        If any of the csv files are not found at the given path.
    """

    path_data = os.path.join(path, "data/interactions.csv")
    path_items = os.path.join(path, "data/items.csv")
    path_users = os.path.join(path, "data/users.csv")

    # logger.info(f"Reading data from csv files {path_data}")
    assert os.path.exists(path_data), f"File {path_data} not found."
    assert os.path.exists(path_items), f"File {path_data} not found."
    assert os.path.exists(path_users), f"File {path_data} not found."

    data = pd.read_csv(path_data)
    data_users = pd.read_csv(path_users)
    data_items = pd.read_csv(path_items)

    data["start_date"] = pd.to_datetime(data["start_date"])
    data["rating"] = np.array(data["rating"].values, dtype=np.float32)

    return data, data_users, data_items


def create_weighted_interaction_matrix(data: pd.DataFrame, alpha=0.01):
    """
    Create a weighted interaction matrix based on the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing interactions between users and items.
    alpha : float, optional
        A hyperparameter used to weight interactions based on recency. Default is 0.01.

    Returns
    -------
    Tuple[pd.DataFrame, sp.coo_matrix]
        The input DataFrame with added columns for days since interaction, weight, and target, and a sparse matrix
        representing the weighted interactions.
    """

    data.loc[:, "target"] = (data["rating"].fillna(0)
                             * 20 + data["progress"]) / 2

    interactions_sparse = sp.coo_matrix(
        (
            data["target"].astype(float),
            (data["user_id"].astype(int), data["item_id"].astype(int)),
        ),
    )
    return data, interactions_sparse

