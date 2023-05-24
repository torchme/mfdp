import os
import pickle
import sys
import requests
from urllib.parse import urlencode
from gpt4all import GPT4All
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)


def download_and_extract_dataset(filename: str):
    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"

    if filename == "interactions.csv":
        url = "https://disk.yandex.ru/d/_ATsGOU_LMzPPg?direct=1"
    elif filename == "items.csv":
        url = "https://disk.yandex.ru/d/sT6izpVyyDZQVA?direct=1"
    elif filename == "users.csv":
        url = "https://disk.yandex.ru/d/PN6yRVPBBJycLg?direct=1"
    elif filename == "als.pickle":
        url = "https://disk.yandex.ru/d/EQ-xiSy6fxCy1Q?direct=1"
    elif filename == "bm25.pickle":
        url = "https://disk.yandex.ru/d/cIPTc4LhgAYLGg?direct=1"

    os.makedirs("src/model", exist_ok=True)
    os.makedirs("src/data", exist_ok=True)

    if filename == "bm25.pickle" or filename == "als.pickle":
        pathfilename = ".src/model/" + filename
    else:
        pathfilename = ".src/data/" + filename

    final_url = base_url + urlencode(dict(public_key=url))
    response = requests.get(final_url)
    download_url = response.json()["href"]

    download_response = requests.get(download_url)
    with open(pathfilename, "wb") as f:
        f.write(download_response.content)


def check_upload():

    if not os.path.isfile(os.path.join(sys.path[1], "./src/data/interactions.csv")):
        download_and_extract_dataset("interactions.csv")

    if not os.path.isfile(os.path.join(sys.path[1], "./src/data/items.csv")):
        download_and_extract_dataset("items.csv")

    if not os.path.isfile(os.path.join(sys.path[1], "./src/data/users.csv")):
        download_and_extract_dataset("users.csv")

    if not os.path.isfile(os.path.join(sys.path[1], "./src/model/als.pickle")):
        download_and_extract_dataset("als.pickle")

    if not os.path.isfile(os.path.join(sys.path[1], "./src/model/bm25.pickle")):
        download_and_extract_dataset("bm25.pickle")


def load_gpt():
    path = os.path.join(sys.path[1], "./src/model/ggml-gpt4all-j-v1.3-groovy.bin")
    status = os.path.isfile(path)
    if not status:
        GPT4All.download_model("ggml-gpt4all-j-v1.3-groovy.bin", "./src/model/")


def load_t5():
    path_modelone = os.path.join(sys.path[1], "./src/model/spiece.model")
    status_one = os.path.isfile(path_modelone)

    path_modeltwo = os.path.join(sys.path[1], "./src/model/pytorch_model.bin")
    status_two = os.path.isfile(path_modeltwo)

    if not status_one or not status_two:
        MODEL_NAME = "cointegrated/rut5-base-multitask"
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

        SAVE_DIRECTORY = ".src/model/"
        tokenizer.save_pretrained(SAVE_DIRECTORY)
        model.save_pretrained(SAVE_DIRECTORY)


def load_file():
    check_upload()
    load_gpt()
    load_t5()


if __name__ == "__main__":
    load_file()
