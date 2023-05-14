import os
import pickle
import sys

import pandas as pd
import streamlit as st

from src.recomend import popular_items
from src.utils import get_user_loggins, read_data


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
    if len(data) < k:
        st.error("The number of data points should be greater than or equal to k.")
        return

    num_data = len(data)
    num_columns = min(k, num_data)

    columns = st.columns(num_columns)
    for i in range(num_columns):
        item = data.iloc[i]

        ## Search for a random photo on Unsplash
        # photo = api.photo.random()

        ## Get the URL of the photo
        # photo_url = photo[0].urls.raw

        # Download the image
        # response = requests.get(photo_url)
        # img = Image.open(BytesIO(response.content))

        ## Resize the image to 400x300 pixels
        # img = img.resize((200, 300))
        ## Display the image
        # columns[i].image(img, width=200)

        # Display the title
        columns[i].write(f'**{item["title"]}**')

        # Display the author, year, and genres
        columns[i].write(f"**Автор:** {item['authors']}")
        columns[i].write(f"**Год:** {item['year']}")
        columns[i].write(f"**Жанры:** {item['genres']}")


intercations, data, data_items = read_data(os.path.join(sys.path[1], "src"))

users = get_user_loggins(data)

query_params = st.experimental_get_query_params()
nickname = query_params.get("nickname", [None])[0]

if nickname is None:
    st.set_page_config(page_title="DontReadMe.com", page_icon="📘", layout="wide")

    row0_1, row0_2, row0_3, row0_4, row0_5 = st.columns((2, 2, 2, 2, 2))

    row0_3.title("Dont:blue[Read]Me :book:")

    row1_1, row1_2, row1_3 = st.columns((2, 2, 2))

    row1_2.subheader(
        ":blue[Cкоро ты узнаешь, что такое настоящая зависимость от чтения.]"
    )

    row2_1, row2_2, row2_3 = st.columns((2, 2, 2))

    nickname = row2_2.text_input("**Nickname**", placeholder="Please enter your id")
    if row2_2.button("Dont click me", type="primary"):
        if int(nickname) in users:
            row2_2.success("Match found!")
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
        ":blue[Cкоро ты узнаешь, что такое настоящая зависимость от чтения.]"
    )

    row2_1, row2_2, row2_3 = st.columns((0.5, 2, 0.5))

    row2_2.divider()
    row2_2.subheader("Похожие книги")
    text = row2_2.text_input("Введите название книги", key="text2")
    button2 = row2_2.button("Найти", type="secondary", key="button2")
    if button2:
        if str(text) in list(data_items["title"].unique()):
            row2_2.success("Match found!")
            id_item = data_items[data_items["title"] == text]["id"]
            with open(os.path.join("./src/model/bm25.pickle"), "rb") as f:
                item_model = pickle.load(f)
            similar_items = item_model.similar_items(id_item, k)
            similar = pd.DataFrame(
                {"col_id": similar_items[0][0], "similarity": similar_items[1][0]}
            )
            items_inv_mapping = dict(enumerate(intercations["item_id"].unique()))
            items_mapping = {v: k for k, v in items_inv_mapping.items()}
            item_titles = pd.Series(
                data_items["title"].values, index=data_items["id"]
            ).to_dict()
            similar["item_id"] = similar["col_id"].map(items_inv_mapping.get)
            similar["title"] = similar["item_id"].map(item_titles.get)
            create_columns_with_data(
                row2_2, k, data=data_items[data_items["title"].isin(similar["title"])]
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
    with open(os.path.join("./src/model/als.pickle"), "rb") as f:
        user_model = pickle.load(f)

    users = user_model.similar_users(int(nickname), N=5)
    users = [i[0] for i in users]

    users_inv_mapping = dict(enumerate(intercations["user_id"].unique()))
    users_mapping = {v: k for k, v in users_inv_mapping.items()}
    item_titles = pd.Series(
        data_items["title"].values, index=data_items["id"]
    ).to_dict()

    user_item_list = []
    for uid in users:
        user_id = users_inv_mapping[uid]
        user_mask = intercations["user_id"] == user_id

        user_items = intercations.loc[user_mask, "item_id"].map(item_titles.get)
        user_item_list.extend(user_items.values)

    create_columns_with_data(
        row2_2, k, data=data_items[data_items["title"].isin(user_item_list)]
    )
