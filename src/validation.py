import os
import pickle
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import wandb
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import ranking_metrics_at_k, train_test_split
from implicit.nearest_neighbours import BM25Recommender

from utils import create_weighted_interaction_matrix, read_data


def split_train_test_users(interactions, num_folds=5):
    """
    Splits the interactions dataframe into train and test sets by unique users using k-fold cross-validation.

    Parameters
    ----------
    interactions : pandas.DataFrame
        The interactions dataframe.
    num_folds : int, optional
        The number of folds for k-fold cross-validation. Default is 5.

    Yields
    -------
    tuple
        A tuple of pandas.DataFrames containing the train and test sets, respectively.
    """
    for random_state in range(num_folds):
        _, sparce_mat = create_weighted_interaction_matrix(interactions)
        train, test = train_test_split(sparce_mat, random_state=random_state)
        yield train, test


def fit_implicit_als(
    train_sparse,
    test_sparse=None,
    iterations=15,
    factors=20,
    regularization=0.1,
    alpha=40,
    log_wandb=True,
    fold=0
):
    """
    Fits an implicit ALS model using the given training sparse matrix.
    Optionally evaluates the model using the given test sparse matrix and logs the results to Weights & Biases.
    If no test sparse matrix is given, the trained model is pickled and saved to disk.

    Parameters
    ----------
    train_sparse : scipy.sparse.csr_matrix
        The training sparse matrix.
    test_sparse : scipy.sparse.csr_matrix, optional
        The test sparse matrix.
    iterations : int, optional
        The number of iterations to run the ALS algorithm. Default is 15.
    factors : int, optional
        The number of latent factors to use in the model. Default is 20.
    regularization : float, optional
        The regularization parameter to use in the ALS algorithm. Default is 0.1.
    alpha : int, optional
        The alpha hyperparameter to use in the confidence matrix of the ALS algorithm. Default is 40.
    log_wandb : bool, optional
        Whether to log the results to Weights & Biases. Default is True.

    Returns
    -------
    None
    """

    params = {
        "factors": factors,
        "regularization": regularization,
        "iterations": iterations,
        "alpha": alpha,
    }
    model = AlternatingLeastSquares(**params)

    if log_wandb:
        run = wandb.init(project="MFDP", name="ALS")
        wandb.config.update(params)

    model.fit(train_sparse)

    if test_sparse is not None:
        for k in [1, 5, 10, 100]:
            metrics = ranking_metrics_at_k(
                model,
                train_sparse,
                test_sparse,
                K=k,
                show_progress=False,
            )
            metrics["k"] = k
            metrics['fold'] = fold
            wandb.log(metrics)
    else:
        with open(os.path.join(sys.path[0], "./model/als.pickle"), "wb") as f:
            pickle.dump(model, f)

        artifact = wandb.Artifact("als-model", type='pickle')
        with artifact.new_file(os.path.join(sys.path[0], "./model/als.pickle"), mode='wb') as f:
            pickle.dump(model, f)
        run.log_artifact(artifact)
        wandb.finish()



def fit_implicit_bm25(
    train_sparse,
    test_sparse=None,
    log_wandb=True,
    fold=0
):
    """
    Fits an implicit ALS model using the given training sparse matrix.
    Optionally evaluates the model using the given test sparse matrix and logs the results to Weights & Biases.
    If no test sparse matrix is given, the trained model is pickled and saved to disk.

    Parameters
    ----------
    train_sparse : scipy.sparse.csr_matrix
        The training sparse matrix.
    test_sparse : scipy.sparse.csr_matrix, optional
        The test sparse matrix.
    log_wandb : bool, optional
        Whether to log the results to Weights & Biases. Default is True.

    Returns
    -------
    None
    """
    run = wandb.init(project="MFDP", name="BM25")

    model = BM25Recommender(K=50, B=0.5)

    model.fit(train_sparse)

    if test_sparse is not None:
        for k in [1, 5, 10, 100]:
            metrics = ranking_metrics_at_k(
                model,
                train_sparse,
                test_sparse,
                K=k,
                show_progress=False,
            )
            metrics["k"] = k
            metrics['fold'] = fold
            wandb.log(metrics)
    else:
        with open(os.path.join(sys.path[0], "./model/bm25.pickle"), "wb") as f:
            pickle.dump(model, f)

        artifact = wandb.Artifact("bm25-model", type='pickle')
        with artifact.new_file(os.path.join(sys.path[0], "./model/bm25.pickle"), mode='wb') as f:
            pickle.dump(model, f)
        run.log_artifact(artifact)
        wandb.finish()


def main(folds=7, validation=False, algorithm="als"):
    intercations, _, _ = read_data(os.path.join(sys.path[0]))

    if validation == True:
        run_no = 0
        for train, test in split_train_test_users(intercations, num_folds=folds):
            
            if algorithm == "als":
                fit_implicit_als(train, test, iterations=25, factors=50, fold=run_no)
            else:
                fit_implicit_bm25(train, test, fold=run_no)
            run_no += 1
    else:
        _, sparce_mat = create_weighted_interaction_matrix(intercations)
        if algorithm == "als":
            fit_implicit_als(sparce_mat, iterations=25, factors=50)
        else:
            fit_implicit_bm25(sparce_mat)


if __name__ == "__main__":
    main(folds=3, algorithm="als", validation=True)
