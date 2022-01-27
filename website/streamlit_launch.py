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
from src.vasp.vasp import vasp_model
from src.Content_Based.content_based import content_based_model

# Data Loading
title_list = pd.read_json('artifacts/items_pu5.json').product_name.values.tolist()


# App declaration
def main():
    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    #page_options = ["Recommender System", "Solution Overview"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    #page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if True:#page_selection == "Recommender System":
        # Header contents
        st.write('### Movie Recommender Engine')
        # st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        #st.image('resources/imgs/Image_header.png', use_column_width=True)

        sys = st.radio("Select an algorithm",
                       ("Collaboratif filtering", "Content Based", 'VASP', "Hybrid"))

        # User-based preferences
        # st.write('Enter Your Favorite Movies')
        # movie_1 = st.selectbox('Fisrt Option',title_list)
        # movie_2 = st.selectbox('Second Option',title_list)
        # movie_3 = st.selectbox('Third Option',title_list)
        # fav_movies = [movie_1,movie_2,movie_3]
        fav_movies = st.multiselect("Enter Your Favorite Movies",
                                    title_list)
        number = st.slider("How much movies you want to be recommended:", 5, 100)

        # Perform top-100 movie recommendation generation
        if sys == "Hybrid":
            st.write("Not implimented")

        if sys == "Collaboratif filtering":
            st.write('not implimented')
        if sys == "Content Based":
            #st.write('Not implimented')
            if st.button("Recommend"):
                try:
                    top_recommendations = content_based_model(fav_movies,number)
                    st.write("We think you'll like:")
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

    # -------------------------------------------------------------------

if __name__ == '__main__':
    main()
