"""Utils."""
from __future__ import annotations

import os

import pandas as pd
import numpy as np

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


def get_coo_matrix(
    intercations: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    weight_col: str = None,
    users_mapping: dict = None,
    items_mapping: dict = None,
) -> sp.coo_matrix:
    """Create a COO sparse matrix from a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the user-item interactions.
    user_col : str, optional
        Name of the user ID column in `df`, by default 'user_id'.
    item_col : str, optional
        Name of the item ID column in `df`, by default 'item_id'.
    weight_col : str, optional
        Name of the weight column in `df`, by default None.
    users_mapping : dict, optional
        A mapping from user IDs to row indices in the resulting matrix, by default None.
    items_mapping : dict, optional
        A mapping from item IDs to column indices in the resulting matrix, by default None.

    Returns
    -------
    sp.coo_matrix
        A sparse COO matrix representing the user-item interactions.
    """
    if weight_col is None:
        weights = np.ones(len(intercations), dtype=np.float32)
    else:
        weights = intercations[weight_col].astype(np.float32)

    interaction_matrix = sp.coo_matrix(
        (
            weights,
            (
                intercations[user_col].map(users_mapping.get),
                intercations[item_col].map(items_mapping.get),
            ),
        )
    )
    return interaction_matrix


def get_mapping(intercations: pd.DataFrame):
    """
    Returns two mappings for user and item IDs to integer indices.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing user and item IDs.

    Returns
    -------
    tuple of two dictionaries
        The first dictionary maps user IDs to integer indices, and the second
        dictionary maps item IDs to integer indices.
    """
    users_inv_mapping = dict(enumerate(intercations["user_id"].unique()))
    users_mapping = {v: k for k, v in users_inv_mapping.items()}

    items_inv_mapping = dict(enumerate(intercations["item_id"].unique()))
    items_mapping = {v: k for k, v in items_inv_mapping.items()}

    return users_mapping, items_mapping


def get_user_loggins(data):
    """Extracts unique user IDs from the given data."""
    users = data['user_id'].unique()
    return list(users)
