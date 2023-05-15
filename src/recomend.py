"""recomend.py"""
from __future__ import annotations
import os, sys
import wandb
import pickle

import torch
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
from transformers import T5ForConditionalGeneration, T5Tokenizer

import pandas as pd

from .utils import create_weighted_interaction_matrix, read_data


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
    mask = data[data["progress"] > threshold_progress][["item_id"]]
    mask = mask.value_counts()

    items_count = pd.DataFrame(
        mask,
        columns=["count"],
    ).sort_index()

    items_name = data_items[["id", "title", "genres"]]
    items_name = items_name.set_index("id")
    items_name = items_name.sort_index()

    items_name.genres = items_name.genres.fillna("Другие жанры")
    items_name.genres = items_name.genres.apply(lambda x: x.split(","))

    if genre is not None:
        items_name = items_name.explode(column="genres")
        items_name = items_name[items_name["genres"] == genre][["title"]]

    count_titles = items_name.merge(
        items_count,
        left_index=True,
        right_on="item_id",
    )

    output = count_titles.sort_values(by="count", ascending=False)
    output = output["title"].values[:n_items]
    return output


def summarize(input_sequences: str):
    """
    Суммаризация текста на основе модели T5 LLM.

    Аргументы:
    text -- str, исходный текст для суммаризации.

    Возвращает:
    summary -- str, суммаризованный текст.
    """
    device = torch.device("cpu")

    MODEL_NAME = "UrukHan/t5-russian-summarization"
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    task_prefix = "Summarize: "
    if type(input_sequences) != list:
        input_sequences = [input_sequences]
    encoded = tokenizer(
        [task_prefix + sequence for sequence in input_sequences],
        padding="longest",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    )

    predicts = model.generate(**encoded.to(device))
    summary = tokenizer.batch_decode(predicts, skip_special_tokens=True)
    return summary[0]


def chat(text, **kwargs):
    """
    Генерация ответа на основе модели T5 LLM.

    Аргументы:
    text -- str, исходный текст вопроса.

    Возвращает:
    answer -- str, ответ на вопрос.
    """
    MODEL_NAME = "cointegrated/rut5-base-multitask"

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    task_prefix = "answer | "
    inputs = tokenizer(task_prefix + text, return_tensors="pt")
    with torch.no_grad():
        hypotheses = model.generate(**inputs, num_beams=5, **kwargs)
    return tokenizer.decode(hypotheses[0], skip_special_tokens=True)


def recomend_als(user_id):
    """
    Рекомендация пользователя на основе модели ALS.
    """
    with open('./src/model/als.pickle', 'rb') as f:
        model = pickle.load(f)
    similar_users = model.similar_users(int(user_id))[0]
    if similar_users == []:
        similar_users = model.similar_users(-1)[0]
    intercations, _, items_data = read_data(os.path.join('./src/'))
    intercations, _ = create_weighted_interaction_matrix(intercations)

    dataset = intercations[intercations['user_id'].isin(similar_users)].sort_values(by='target', ascending=False)['item_id']
    items_data = items_data[items_data['id'].isin(dataset)]
    return items_data
