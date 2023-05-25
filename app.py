import os
import sys
import streamlit as st
import requests

from src.recomend import chat, popular_items, recomend_als, recomend_bm25, summarize
from src.utils import read_data
from src.load_data import load_file

from streamlit_lottie import st_lottie
from trubrics.integrations.streamlit import FeedbackCollector


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def create_product_card(st, k: int, data: list):
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

        columns[i].write(f'**{item["title"]}**')

        columns[i].write(f"**Автор:** {item['authors']}")
        columns[i].write(f"**Год:** {item['year']}")
        columns[i].write(f"**Жанры:** {item['genres']}")
        # with columns[i]:
        #    collector = FeedbackCollector()
        #    collector.st_feedback(
        #        feedback_type="thumbs"
        #    )


def main():
    intercations, data, data_items = read_data(os.path.join(sys.path[1], "src"))

    users = data["user_id"].unique().tolist()

    query_params = st.experimental_get_query_params()
    nickname = query_params.get("nickname", [None])[0]

    if nickname is None:
        st.set_page_config(page_title="DontReadMe.com", page_icon="📘", layout="wide")

        _, _, row0_3, _, _ = st.columns((2, 2, 2, 2, 2))

        with row0_3:
            st_lottie(
                load_lottieurl(
                    "https://assets8.lottiefiles.com/packages/lf20_1a8dx7zj.json"
                ),
                key="user",
            )

        row0_3.title("Dont:blue[Read]Me :book:")

        _, row1_2, _ = st.columns((2, 2, 2))

        row1_2.subheader(
            ":blue[Cкоро ты узнаешь, что такое настоящая зависимость от чтения.]",
        )

        _, row2_2, _ = st.columns((2, 2, 2))

        nickname = row2_2.text_input(
            "**Nickname**",
            placeholder="Введите ваш никнейм",
        )
        if row2_2.button("Войти", type="primary"):
            if int(nickname) in users:
                st.experimental_set_query_params(nickname=int(nickname))
                # st.experimental_rerun()
            else:
                row2_2.error("Match not found!")
        row2_2.divider()
        row2_2.header("О проекте")
        row2_2.write('''
        Проект **DontReadMe** - это рекомендательная система книг, которая предназначена для решения проблемы долгого чтения. Она позволяет :blue[сократить время], которое требуется на подбор литературы, а также добавляет функции по :blue[суммаризации текста], что упрощает процесс чтения и позволяет быстрее усваивать информацию.

        **Основная цель проекта** - это предоставление пользователю :blue[персонализированных] рекомендаций книг, основанных на его предпочтениях и интересах. Для этого система использует алгоритмы машинного обучения, которые анализируют историю чтения пользователя, его оценки и отзывы, а также другие параметры, чтобы определить наиболее подходящие книги для него.

        Кроме того, система также предоставляет возможность суммаризации текста, что упрощает процесс чтения и позволяет быстрее усваивать информацию. Эта функция основана на алгоритмах обработки естественного языка, которые анализируют текст и выделяют наиболее важные и значимые его части.

        В целом, проект **DontReadMe** - это :blue[инновационный подход] к чтению книг, который позволяет ускорить процесс чтения, сделать его более удобным и эффективным, а также получать персонализированные рекомендации книг, которые соответствуют интересам и предпочтениям пользователя.
        ''')
    else:
        st.set_page_config(page_title="DontReadMe.com", page_icon="📘", layout="wide")

        k = 5

        _, _, row0_3, _, _ = st.columns((2, 2, 2, 2, 2))

        row0_3.title("Dont:blue[Read]Me :book:")

        _, row1_2, _ = st.columns((2, 2, 2))

        row1_2.subheader(
            ":blue[Cкоро ты узнаешь, что такое настоящая зависимость от чтения.]",
        )

        _, row2_2, _ = st.columns((0.5, 2, 0.5))

        # // bm25
        row2_2.divider()
        row2_2.subheader("Похожие книги")
        text = row2_2.text_input("Введите название книги", key="text2")
        button2 = row2_2.button("Найти", type="secondary", key="button2")
        if button2:
            if str(text) in list(data_items["title"].unique()):
                row2_2.success("Match found!")
                id_item = data_items[data_items["title"] == text]["id"].values[0]
                books = recomend_bm25(id_item)

                create_product_card(
                    row2_2,
                    k,
                    data=books,
                )
            else:
                row2_2.error("Match not found!")

        row2_2.divider()

        # // Cold Start
        row2_2.subheader("Популярное")

        items = popular_items(data=intercations, data_items=data_items)
        data = data_items[data_items["title"].isin(items)]
        create_product_card(row2_2, k, data=data)

        # // ALS
        row2_2.divider()

        row2_2.subheader("Рекомендации")
        data_items = recomend_als(nickname)

        create_product_card(
            row2_2,
            k,
            data=data_items,
        )

        # // Summarize
        _, row3_2, _ = st.columns((0.5, 2, 0.5))
        row3_2.divider()
        row3_2.subheader("Кратко-бот")
        text_to_summarize = row3_2.text_input("Введите описание книги")
        if text_to_summarize:
            row3_2.write(summarize(text_to_summarize))

        # // Chat-bot
        row3_2.divider()
        row3_2.subheader("Чат-бот")
        text_to_asnwer = row3_2.text_input("Можете спросить совет у бота", key="other")
        # if text_to_asnwer:
        #    row3_2.write(chat(text_to_summarize))


if __name__ == "__main__":
    load_file()
    main()
