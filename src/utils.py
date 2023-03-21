"""Utils."""
from __future__ import annotations

import os

import pandas as pd
from loguru import logger


def read_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read data from csv files.

    Parameters
    ----------
    path : str
        path to csv files

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        output dataframes
    """
    path_data = os.path.join(path, 'data/interactions.csv')
    path_items = os.path.join(path, 'data/items.csv')
    path_users = os.path.join(path, 'data/users.csv')

    logger.info('Reading data from csv files.')
    assert os.path.exists(path_data), f'File {path_data} not found.'
    assert os.path.exists(path_items), f'File {path_data} not found.'
    assert os.path.exists(path_users), f'File {path_data} not found.'

    data = pd.read_csv(path_data)
    data_users = pd.read_csv(path_users)
    data_items = pd.read_csv(path_items)

    return data, data_users, data_items
