import os
import random
import time
from os.path import split, join
from sys import path
import requests

path.insert(0, join(split(__file__)[0], '..'))

# Streamlit dependencies
import streamlit as st

st.set_page_config(
    page_title="ENSIAS",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)
# Data handling dependencies
import pandas as pd
import numpy as np
import json

# Custom Libraries

print('import vasp')
from src.vasp.vasp import vasp_model

print('import content')
from src.Content_Based.content_based import content_based_model, metadata

print('import collaborative')
# from src.collaborative_filtering.utils import recommend as knn_recommender,fuzzy_matching
print('end import')

# Data Loading
title_list = pd.read_json('artifacts/items_pu5.json').product_name.values.tolist()


def hybrid(l1, l2):
    inter = [ e for e in l1 if e in l2]
    l1 = [ e for e in l1 if e not in inter]
    l2 = [e for e in l2 if e not in inter]
    return inter+l1+l2

id = random.randint(0,10000000)



# App declaration
def main():
    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    # page_options = ["Recommender System", "Solution Overview"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    # page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if True:  # page_selection == "Recommender System":
        # Header contents
        st.write('### Movie Recommender Engine')
        # st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        # st.image('resources/imgs/Image_header.png', use_column_width=True)

        sys = st.radio("Select an algorithm",
                       ("Collaboratif filtering", "Content Based", 'VASP', "Hybrid"))

        fav_movies = st.multiselect("Enter Your Favorite Movies",
                                    title_list)
        number = st.slider("How much movies you want to be recommended:", 5, 100)

        top_recommendations = []
        # Perform top-100 movie recommendation generation
        if sys == "Hybrid":
            # st.write('Not implimented')
            if st.button("Recommend"):
                try:
                    top_recommendations1 = knn_recommender(fav_movies, number)

                    elem = [title_list.index(i) + 1 for i in fav_movies]
                    vec = np.asarray([(i in elem) * 1 for i in range(len(title_list) + 1)]).reshape(1, -1)
                    predictions = vasp_model(vec)
                    # predictions = np.random.random(size = 1000)
                    top10 = reversed(predictions.argsort()[-number:])
                    top_recommendations2 = [title_list[i - 1] for i in top10]

                    top_recommendations = hybrid(top_recommendations2,top_recommendations1)[:number]
                    st.write("We think you'll like:")
                    st.write(top_recommendations)
                    st.download_button(
                        label="Download the results",
                        data=json.dumps(top_recommendations),
                        file_name='large_df.json',
                        mime='text/json',
                    )
                except:
                    st.error("Oops! Looks like something went wrong Try Again ...")

        if sys == "Collaboratif filtering":
            # st.write('Not implimented')
            if st.button("Recommend"):
                try:
                    top_recommendations = knn_recommender(fav_movies, number)
                    st.write("We think you'll like:")
                    st.write(top_recommendations)
                    st.download_button(
                        label="Download the results",
                        data=json.dumps(top_recommendations),
                        file_name='large_df.json',
                        mime='text/json',
                    )
                except:
                    st.error("Oops! Looks like something went wrong Try Again ...")

        if sys == "Content Based":

            # metadata
            # fav_movies = st.multiselect("Enter Your Favorite Movies",
            #                             metadata.original_title.values.tolist())
            # st.write('Not implimented')


            if st.button("Recommend"):
                try:
                    new_list = [fuzzy_matching(metadata.original_title.values.tolist(), i) for i in fav_movies]
                    new_list = [i for i in new_list if i!=None]
                    print('###############',fav_movies[0],new_list)
                    top_recommendations = content_based_model(new_list, number).values.tolist()

                    # top_recommendations = [fuzzy_matching(title_list, i) for i in top_recommendations]
                    # top_recommendations = [i for i in top_recommendations if i != None]

                    st.write("We think you'll like:")
                    st.write(top_recommendations)
                    st.download_button(
                        label="Download the results",
                        data=json.dumps(top_recommendations),
                        file_name='large_df.json',
                        mime='text/json',
                    )
                except:
                    st.error("Oops! Looks like something went wrong Try Again ...")

        if sys == "VASP":
            # st.write("The current model")
            if st.button("Recommend"):
                try:
                    elem = [title_list.index(i) + 1 for i in fav_movies]
                    vec = np.asarray([(i in elem) * 1 for i in range(len(title_list) + 1)]).reshape(1, -1)
                    predictions = vasp_model(vec)
                    # predictions = np.random.random(size = 1000)
                    top10 = reversed(predictions.argsort()[-number:])
                    top_recommendations = [title_list[i - 1] for i in top10]
                    st.write("We think you'll like:")
                    st.write(top_recommendations)
                    st.download_button(
                        label="Download the results",
                        data=json.dumps(top_recommendations),
                        file_name='large_df.json',
                        mime='text/json',
                    )
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")
        # save recommendation
        if fav_movies!=[]:
            with open("dataset.csv", 'a') as f:
                f.write(','.join([ str(time.time()) , str(id),str(len(fav_movies)), str(len(top_recommendations))]+fav_movies+top_recommendations)+'\n')
            os.system('hadoop fs –put dataset.csv')
    # -------------------------------------------------------------------


if __name__ == '__main__':
    main()
