import os
import sys

import pickle
import numpy as np
import pandas as pd
import wandb
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.evaluation import ranking_metrics_at_k
from implicit.nearest_neighbours import BM25Recommender, TFIDFRecommender
from more_itertools import pairwise

from utils import get_coo_matrix, get_mapping, read_data


class TimeRangeSplit:
    """From https://www.kaggle.com/code/sharthz23/implicit-lightfm."""

    def __init__(
        self,
        start_date,
        end_date=None,
        freq="D",
        periods=None,
        tz=None,
        normalize=False,
        closed=None,
        train_min_date=None,
        filter_cold_users=True,
        filter_cold_items=True,
        filter_already_seen=True,
    ):
        """Initialize TimeRangeSplit."""
        self.start_date = start_date
        if end_date is None and periods is None:
            raise ValueError(
                "Either 'end_date' or 'periods' must be non-zero, not both at the same time.",
            )

        self.end_date = end_date
        self.freq = freq
        self.periods = periods
        self.tz = tz
        self.normalize = normalize
        self.closed = closed
        self.train_min_date = pd.to_datetime(train_min_date, errors="raise")
        self.filter_cold_users = filter_cold_users
        self.filter_cold_items = filter_cold_items
        self.filter_already_seen = filter_already_seen

        self.date_range = pd.date_range(
            start=start_date,
            end=end_date,
            freq=freq,
            periods=periods,
            tz=tz,
            normalize=normalize,
            closed=closed,
        )

        self.max_n_splits = max(0, len(self.date_range) - 1)
        if self.max_n_splits == 0:
            raise ValueError("Provided parametrs set an empty date range.")

    def split(
        self,
        df,
        user_column="user_id",
        item_column="item_id",
        datetime_column="date",
        fold_stats=False,
    ):
        """Split the dataset into training and test sets."""
        df_datetime = df[datetime_column]
        if self.train_min_date is not None:
            train_min_mask = df_datetime >= self.train_min_date
        else:
            train_min_mask = df_datetime.notnull()

        date_range = self.date_range[
            (self.date_range >= df_datetime.min())
            & (self.date_range <= df_datetime.max())
        ]

        for start, end in pairwise(date_range):
            fold_info = {"Start date": start, "End date": end}
            train_mask = train_min_mask & (df_datetime < start)
            train_idx = df.index[train_mask]
            if fold_stats:
                fold_info["Train"] = len(train_idx)

            test_mask = (df_datetime >= start) & (df_datetime < end)
            test_idx = df.index[test_mask]

            if self.filter_cold_users:
                new = np.setdiff1d(
                    df.loc[test_idx, user_column].unique(),
                    df.loc[train_idx, user_column].unique(),
                )
                new_idx = df.index[test_mask & df[user_column].isin(new)]
                test_idx = np.setdiff1d(test_idx, new_idx)
                test_mask = df.index.isin(test_idx)
                if fold_stats:
                    fold_info["New users"] = len(new)
                    fold_info["New users interactions"] = len(new_idx)

            if self.filter_cold_items:
                new = np.setdiff1d(
                    df.loc[test_idx, item_column].unique(),
                    df.loc[train_idx, item_column].unique(),
                )
                new_idx = df.index[test_mask & df[item_column].isin(new)]
                test_idx = np.setdiff1d(test_idx, new_idx)
                test_mask = df.index.isin(test_idx)
                if fold_stats:
                    fold_info["New items"] = len(new)
                    fold_info["New items interactions"] = len(new_idx)

            if self.filter_already_seen:
                user_item = [user_column, item_column]
                train_pairs = df.loc[train_idx, user_item].set_index(user_item).index
                test_pairs = df.loc[test_idx, user_item].set_index(user_item).index
                intersection = train_pairs.intersection(test_pairs)
                test_idx = test_idx[~test_pairs.isin(intersection)]
                # test_mask = rd.df.index.isin(test_idx)
                if fold_stats:
                    fold_info["Known interactions"] = len(intersection)

            if fold_stats:
                fold_info["Test"] = len(test_idx)

            yield (train_idx, test_idx, fold_info)

    def get_n_splits(self, df, datetime_column="date"):
        """Get n splits."""
        df_datetime = df[datetime_column]
        if self.train_min_date is not None:
            df_datetime = df_datetime[df_datetime >= self.train_min_date]

        date_range = self.date_range[
            (self.date_range >= df_datetime.min())
            & (self.date_range <= df_datetime.max())
        ]

        return max(0, len(date_range) - 1)


def validation_bpr(folds_with_stats, df, users_mapping, items_mapping, k=10):
    """Run k-fold cross-validation for Bayesian Personalized Ranking (BPR).

    For each fold, fit a BPR model on the training set, and evaluate it on
    the test set using ranking metrics@k. The metrics are printed to the
    console.

    Parameters
    ----------
    folds_with_stats : list of tuple of 3 arrays
        A list of folds, where each fold is represented as a tuple of 3 arrays:
        the training indices, test indices, and a dictionary of fold statistics.
    df : pandas.DataFrame
        The input DataFrame containing the user-item interactions.

    Returns
    -------
    None
    """
    wandb.init(project="MFDP", name="BPR")

    run_no = 0
    for train_idx, test_idx, _ in folds_with_stats:
        run_no += 1

        train = df.loc[train_idx]
        test = df.loc[test_idx]

        train_mat = get_coo_matrix(
            train,
            users_mapping=users_mapping,
            items_mapping=items_mapping,
        ).tocsr()

        test_mat = get_coo_matrix(
            test,
            users_mapping=users_mapping,
            items_mapping=items_mapping,
        ).tocsr()

        model = BayesianPersonalizedRanking(factors=32, iterations=30)
        model.fit(train_mat.T, show_progress=False)

        metrics = ranking_metrics_at_k(
            model,
            train_mat.T,
            test_mat.T,
            K=k,
            show_progress=False,
        )

        metrics["fold"] = run_no

        wandb.log(metrics)

    data_mat = get_coo_matrix(
        df,
        users_mapping=users_mapping,
        items_mapping=items_mapping,
    ).tocsr()

    model = BayesianPersonalizedRanking(factors=32, iterations=30)
    model.fit(data_mat.T, show_progress=False)

    with open(os.path.join(sys.path[0], "./model/bpr.pickle"), "wb") as f:
        pickle.dump(model, f)

    artifact = wandb.Artifact("model", type="model")
    with artifact.new_file(
        os.path.join(sys.path[0], "./model/bpr.pickle"), mode="wb"
    ) as file:
        pickle.dump(model, file)

    wandb.log_artifact(artifact)

    wandb.finish()


def validation_als(folds_with_stats, df, users_mapping, items_mapping, k=10):
    """
    Perform k-fold cross-validation on the ALS model.

    Parameters
    ----------
    folds_with_stats : list
        A list of (train_idx, test_idx, stats) tuples, where train_idx and test_idx
        are the indices of the training and test sets, respectively, and stats is a
        dictionary containing statistics about the fold.
    df : pandas.DataFrame
        The interaction matrix, with columns "user_id", "item_id", and "weight".

    Returns
    -------
    None
    """
    wandb.init(project="MFDP", name="ALS")

    run_no = 0
    for train_idx, test_idx, _ in folds_with_stats:
        run_no += 1

        train = df.loc[train_idx]
        test = df.loc[test_idx]

        train_mat = get_coo_matrix(
            train,
            users_mapping=users_mapping,
            items_mapping=items_mapping,
        ).tocsr()

        test_mat = get_coo_matrix(
            test,
            users_mapping=users_mapping,
            items_mapping=items_mapping,
        ).tocsr()

        model = AlternatingLeastSquares(factors=32, iterations=30)
        model.fit(train_mat.T, show_progress=False)

        metrics = ranking_metrics_at_k(
            model,
            train_mat.T,
            test_mat.T,
            K=k,
            show_progress=False,
        )
        metrics["fold"] = run_no

        wandb.log(metrics)

    data_mat = get_coo_matrix(
        df,
        users_mapping=users_mapping,
        items_mapping=items_mapping,
    ).tocsr()

    model = AlternatingLeastSquares(factors=32, iterations=30)
    model.fit(data_mat.T, show_progress=False)

    with open(os.path.join(sys.path[0], "./model/als.pickle"), "wb") as f:
        pickle.dump(model, f)

    artifact = wandb.Artifact("model", type="model")
    with artifact.new_file(
        os.path.join(sys.path[0], "./model/als.pickle"), mode="wb"
    ) as file:
        pickle.dump(model, file)

    wandb.log_artifact(artifact)

    wandb.finish()


def validation_tfidf(folds_with_stats, df, users_mapping, items_mapping, k=10):
    """
    Train and evaluate a TF-IDF recommender on cross-validation folds.

    Parameters
    ----------
    folds_with_stats : List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]
        List of tuples where each tuple contains train indices, test indices, and some
        stats about the fold.
    df : pd.DataFrame
        The ratings dataframe.
    users_mapping : Dict
        Mapping from user ids to integers.
    items_mapping : Dict
        Mapping from item ids to integers.

    Returns
    -------
    None
    """
    wandb.init(project="MFDP", name="TF-IDF")

    run_no = 0
    for train_idx, test_idx, _ in folds_with_stats:
        run_no += 1

        train = df.loc[train_idx]
        test = df.loc[test_idx]

        train_mat = get_coo_matrix(
            train,
            users_mapping=users_mapping,
            items_mapping=items_mapping,
        ).tocsr()

        test_mat = get_coo_matrix(
            test,
            users_mapping=users_mapping,
            items_mapping=items_mapping,
        ).tocsr()

        model = TFIDFRecommender(K=k)
        model.fit(train_mat, show_progress=False)

        metrics = ranking_metrics_at_k(
            model,
            train_mat,
            test_mat,
            K=k,
            show_progress=False,
        )
        metrics["fold"] = run_no

        wandb.log(metrics)

    data_mat = get_coo_matrix(
        df,
        users_mapping=users_mapping,
        items_mapping=items_mapping,
    ).tocsr()

    model = TFIDFRecommender(K=k)
    model.fit(data_mat, show_progress=False)

    with open(os.path.join(sys.path[0], "./model/tfidf.pickle"), "wb") as f:
        pickle.dump(model, f)

    artifact = wandb.Artifact("model", type="model")
    with artifact.new_file(
        os.path.join(sys.path[0], "./model/tfidf.pickle"), mode="wb"
    ) as file:
        pickle.dump(model, file)

    wandb.log_artifact(artifact)

    wandb.finish()


def validation_bm25(folds_with_stats, df, users_mapping, items_mapping, k=10):
    """
    Validate TF-IDF recommender using k-fold cross-validation and log results to WandB.

    Parameters
    ----------
    folds_with_stats : list
        List of tuples containing train and test indices and fold statistics.
    df : pd.DataFrame
        Input dataframe containing user-item interaction data.
    users_mapping : dict
        Dictionary mapping user IDs to integer indices.
    items_mapping : dict
        Dictionary mapping item IDs to integer indices.

    Returns
    -------
    None
    """

    wandb.init(project="MFDP", name="BM25")

    run_no = 0
    for train_idx, test_idx, _ in folds_with_stats:
        run_no += 1

        train = df.loc[train_idx]
        test = df.loc[test_idx]

        train_mat = get_coo_matrix(
            train,
            users_mapping=users_mapping,
            items_mapping=items_mapping,
        ).tocsr()

        test_mat = get_coo_matrix(
            test,
            users_mapping=users_mapping,
            items_mapping=items_mapping,
        ).tocsr()

        model = BM25Recommender(K=k)
        model.fit(train_mat, show_progress=False)

        metrics = ranking_metrics_at_k(
            model,
            train_mat,
            test_mat,
            K=k,
            show_progress=False,
        )
        metrics["fold"] = run_no

        wandb.log(metrics)

    data_mat = get_coo_matrix(
        df,
        users_mapping=users_mapping,
        items_mapping=items_mapping,
    ).tocsr()

    model = BM25Recommender(K=k)
    model.fit(data_mat, show_progress=False)

    with open(os.path.join(sys.path[0], "./model/bm25.pickle"), "wb") as f:
        pickle.dump(model, f)

    artifact = wandb.Artifact("model", type="model")
    with artifact.new_file(
        os.path.join(sys.path[0], "./model/bm25.pickle"), mode="wb"
    ) as file:
        pickle.dump(model, file)

    wandb.log_artifact(artifact)

    wandb.finish()


def main(folds=7, k=10):
    intercations, _, _ = read_data(os.path.join(sys.path[0]))

    last_date = intercations["start_date"].max().normalize()
    start_date = last_date - pd.Timedelta(days=folds)

    cv = TimeRangeSplit(start_date=start_date, periods=folds + 1)

    folds_with_stats = list(
        cv.split(
            intercations,
            user_column="user_id",
            item_column="item_id",
            datetime_column="start_date",
            fold_stats=True,
        ),
    )
    wandb.init(project="MFDP", name="validation")
    folds_info_with_stats = pd.DataFrame([info for _, _, info in folds_with_stats])
    folds_info_with_stats.to_csv(os.path.join(sys.path[0], "./data/folds_info.csv"))
    wb_fold_info = wandb.Table(dataframe=folds_info_with_stats)
    wandb.log({"fold info": wb_fold_info})
    wandb.finish()

    users_mapping, items_mapping = get_mapping(intercations)

    validation_bpr(folds_with_stats, intercations, users_mapping, items_mapping, k)

    validation_als(folds_with_stats, intercations, users_mapping, items_mapping, k)

    validation_tfidf(folds_with_stats, intercations, users_mapping, items_mapping, k)

    validation_bm25(folds_with_stats, intercations, users_mapping, items_mapping, k)


if __name__ == "__main__":
    main(folds=7, k=50)
