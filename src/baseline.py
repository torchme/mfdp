"""Baseline."""
from __future__ import annotations

import pandas as pd


def popular_items(
    data: pd.DataFrame,
    data_items: pd.DataFrame,
    genre: str | None = None,
    threshold_progress: int = 40,
    n_items: int = 10,
):
    """Recomends the top n popular items for a given genre.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing the user-item interactions
    df_items : pd.DataFrame
        dataframe containing the items
    genre : str
        genre of the items to be recommended
    threshold_progress : int, optional
        threshold of progress items, by default 40
    n : int, optional
        count items to be return, by default 10

    Returns
    -------
    output: np.ndarray
        the top n popular items for a given genre
    """
    mask = data[data['progress'] > threshold_progress][['item_id']]
    mask = mask.value_counts()

    items_count = pd.DataFrame(
        mask,
        columns=['count'],
    ).sort_index()

    items_name = data_items[['id', 'title', 'genres']]
    items_name = items_name.set_index('id')
    items_name = items_name.sort_index()

    items_name.genres = items_name.genres.fillna('Другие жанры')
    items_name.genres = items_name.genres.apply(lambda x: x.split(','))

    if genre is not None:
        items_name = items_name.explode(column='genres')
        items_name = items_name[items_name['genres'] == genre][['title']]

    count_titles = items_name.merge(
        items_count,
        left_index=True,
        right_on='item_id',
    )

    output = count_titles.sort_values(by='count', ascending=False)
    output = output['title'].values[:n_items]
    return output
