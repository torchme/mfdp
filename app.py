import os
import pickle
import sys
import requests
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from src.recomend import chat, popular_items, recomend_als, summarize
from src.utils import read_data


def download_and_extract_dataset(filename: str):
    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
    # Установить путь к файлу и URL для загрузки датасета
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
        pathfilename = "src/model/" + filename
    else:
        pathfilename = "src/data/" + filename

    # Получаем загрузочную ссылку
    final_url = base_url + urlencode(dict(public_key=url))
    response = requests.get(final_url)
    download_url = response.json()["href"]

    download_response = requests.get(download_url)
    with open(pathfilename, "wb") as f:  # Здесь укажите нужный путь к файлу
        f.write(download_response.content)


def create_columns_with_data(st, k: int, data: list):
    """
    Creates k columns with data in Streamlit.

    Parameters
    ----------
    st : streamlit
        The Streamlit library.
    k : int
        The number of columns to create.
    data : list
        The list of dictionaries containing the data.
    """
    if len(data) < k and len(data) != 0:
        # st.error("The number of data points should be greater than or equal to k.")
        k = len(data)
    elif len(data) == 0:
        st.error("The number of data points should be greater than or equal to k.")
        return

    num_data = len(data)
    num_columns = min(k, num_data)

    columns = st.columns(num_columns)
    for i in range(num_columns):
        item = data.iloc[i]

        # Search for a random photo on Unsplash
        # photo = api.photo.random()

        # Get the URL of the photo
        # photo_url = photo[0].urls.raw

        # Download the image
        # response = requests.get(photo_url)
        # img = Image.open(BytesIO(response.content))

        # Resize the image to 400x300 pixels
        # img = img.resize((200, 300))
        # Display the image
        # columns[i].image(img, width=200)

        # Display the title
        columns[i].write(f'**{item["title"]}**')

        # Display the author, year, and genres
        columns[i].write(f"**Автор:** {item['authors']}")
        columns[i].write(f"**Год:** {item['year']}")
        columns[i].write(f"**Жанры:** {item['genres']}")


if not os.path.isfile(os.path.join(sys.path[1], "src/data/interactions.csv")):
    download_and_extract_dataset("interactions.csv")

if not os.path.isfile(os.path.join(sys.path[1], "src/data/items.csv")):
    download_and_extract_dataset("items.csv")

if not os.path.isfile(os.path.join(sys.path[1], "src/data/users.csv")):
    download_and_extract_dataset("users.csv")

if not os.path.isfile(os.path.join(sys.path[1], "src/model/als.pickle")):
    download_and_extract_dataset("als.pickle")

if not os.path.isfile(os.path.join(sys.path[1], "src/model/bm25.pickle")):
    download_and_extract_dataset("bm25.pickle")


intercations, data, data_items = read_data(os.path.join(sys.path[1], "src"))

users = data["user_id"].unique().tolist()

query_params = st.experimental_get_query_params()
nickname = query_params.get("nickname", [None])[0]

if nickname is None:
    st.set_page_config(page_title="DontReadMe.com", page_icon="📘", layout="wide")

    row0_1, row0_2, row0_3, row0_4, row0_5 = st.columns((2, 2, 2, 2, 2))

    row0_3.title("Dont:blue[Read]Me :book:")

    row1_1, row1_2, row1_3 = st.columns((2, 2, 2))

    row1_2.subheader(
        ":blue[Cкоро ты узнаешь, что такое настоящая зависимость от чтения.]",
    )

    row2_1, row2_2, row2_3 = st.columns((2, 2, 2))

    nickname = row2_2.text_input("**Nickname**", placeholder="Please enter nickname")
    if row2_2.button("Dont click me", type="primary"):
        if int(nickname) in users:
            st.experimental_set_query_params(nickname=int(nickname))
            # st.experimental_rerun()
        else:
            row2_2.error("Match not found!")
else:
    st.set_page_config(page_title="DontReadMe.com", page_icon="📘", layout="wide")

    k = 5

    row0_1, row0_2, row0_3, row0_4, row0_5 = st.columns((2, 2, 2, 2, 2))

    row0_3.title("Dont:blue[Read]Me :book:")

    row1_1, row1_2, row1_3 = st.columns((2, 2, 2))

    row1_2.subheader(
        ":blue[Cкоро ты узнаешь, что такое настоящая зависимость от чтения.]",
    )

    row2_1, row2_2, row2_3 = st.columns((0.5, 2, 0.5))

    row2_2.divider()
    row2_2.subheader("Похожие книги")
    text = row2_2.text_input("Введите название книги", key="text2")
    button2 = row2_2.button("Найти", type="secondary", key="button2")
    if button2:
        if str(text) in list(data_items["title"].unique()):
            row2_2.success("Match found!")
            id_item = data_items[data_items["title"] == text]["id"].values[0]

            with open(os.path.join("./src/model/bm25.pickle"), "rb") as f:
                item_model = pickle.load(f)
            similar_items = item_model.similar_items(id_item)[0][1:]
            books = data_items[data_items["id"].isin(similar_items)]

            create_columns_with_data(
                row2_2,
                k,
                data=books,
            )
        else:
            row2_2.error("Match not found!")

    row2_2.divider()

    row2_2.subheader("Популярное")

    items = popular_items(data=intercations, data_items=data_items)
    data = data_items[data_items["title"].isin(items)]
    create_columns_with_data(row2_2, k, data=data)

    row2_2.divider()

    row2_2.subheader("Рекомендации")
    data_items = recomend_als(nickname)

    create_columns_with_data(
        row2_2,
        k,
        data=data_items,
    )

    row3_1, row3_2, row3_3 = st.columns((0.5, 2, 0.5))
    row3_2.divider()
    row3_2.subheader("Кратко-бот")
    text_to_summarize = row3_2.text_input("Введите описание книги")
    if text_to_summarize:
        row3_2.write(summarize(text_to_summarize))

    row3_2.divider()
    row3_2.subheader("Чат-бот")
    text_to_asnwer = row3_2.text_input("Введите описание книги", key='other')
    if text_to_asnwer:
        row3_2.write(chat(text_to_summarize))
