"""recomend.py"""
from __future__ import annotations

import os
import pickle

import pandas as pd
import torch
from gpt4all import GPT4All
from transformers import (
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5TokenizerFast,
)

from .utils import create_weighted_interaction_matrix, read_data


def popular_items(
    data: pd.DataFrame,
    data_items: pd.DataFrame,
    genre: str | None = None,
    threshold_progress: int = 40,
    n_items: int = 10,
):
    """
    Recomends the top n popular items for a given genre.

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
    """Summarizes the input text

    Parameters
    ----------
    input_sequences : str
        The input text to be summarized

    Returns
    -------
    str
        summarized text
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


def chat_t5(text, **kwargs):
    """
    Initializes the chatbot model and returns the model name.

    Parameters
    ----------
    text : str
        The input text to be processed by the chatbot model.
    **kwargs : dict
        Optional keyword arguments to be passed to the chatbot model.

    Returns
    -------
    str
        The name of the chatbot model.
    """

    MODEL_NAME = "cointegrated/rut5-base-multitask"

    tokenizer = T5Tokenizer.from_pretrained("src/model/")
    model = T5ForConditionalGeneration.from_pretrained("src/model/")
    task_prefix = "Chat: "
    inputs = tokenizer(task_prefix + text, return_tensors="pt")
    with torch.no_grad():
        hypotheses = model.generate(**inputs, num_beams=5, **kwargs)
    return tokenizer.decode(hypotheses[0], skip_special_tokens=True)


def chat(text):
    """
    A function that takes in a text message as a parameter and returns AI-generated text responses using GPT4All model.

    Parameters
    ----------
    text : str
        A string representing the input text message.

    Returns
    -------
    list
        A list of AI-generated text responses.

    Example
    -------
    >>> chat("Hello, how are you?")
    ["I'm fine, thank you. How about yourself?"]
    """

    gptj = GPT4All("ggml-gpt4all-j-v1.3-groovy", "./src/model/")
    messages = [{"role": "user", "content": text}]

    return gptj.chat_completion(messages)["choices"][0]["content"]


def recomend_als(user_id):
    """Recommends books using the ALS algorithm.

    Parameters
    ----------
    user_id : int
        The ID of the user to find similar books to.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing information about the similar books.
    """
    with open("./src/model/als.pickle", "rb") as f:
        model = pickle.load(f)
    similar_users = model.similar_users(int(user_id))[0]
    if similar_users == []:
        similar_users = model.similar_users(-1)[0]
    intercations, _, items_data = read_data(os.path.join("./src/"))
    intercations, _ = create_weighted_interaction_matrix(intercations)

    dataset = intercations[intercations["user_id"].isin(similar_users)].sort_values(
        by="target",
        ascending=False,
    )["item_id"]
    items_data = items_data[items_data["id"].isin(dataset)]
    return items_data


def recomend_bm25(item_id):
    """
    Recommends books similar to a given book ID using the BM25 algorithm.

    Parameters
    ----------
    item_id : int
        The ID of the book to find similar books to.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing information about the similar books.
    """

    _, _, items_data = read_data(os.path.join("./src/"))

    with open(os.path.join("./src/model/bm25.pickle"), "rb") as f:
        item_model = pickle.load(f)
    similar_items = item_model.similar_items(int(item_id))[0][1:]
    books = items_data[items_data["id"].isin(similar_items)]
    return books
