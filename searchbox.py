import enum
import logging
import random
from time import time
from typing import List, Tuple

import requests
import streamlit as st
import pandas as pd

from streamlit_searchbox import st_searchbox

logging.getLogger("streamlit_searchbox").setLevel(logging.DEBUG)

st.set_page_config(layout="wide")



@st.cache_data
def load_dataset():
    new_df = pd.read_csv('tmdb_full.csv')
    new_df.dropna(subset=['poster_path'], inplace=True)
    new_df.dropna(subset=['release_date'], inplace=True)
    new_df.dropna(subset=['imdb_id'], inplace=True)
    new_df.drop_duplicates('imdb_id', inplace=True)
    new_df['release_date'] = pd.to_datetime(new_df['release_date'])
    new_df['release_date_year'] = new_df['release_date'].dt.year.astype('int')
    new_df['genres'].fillna('No Genre', inplace=True)

    return new_df

@st.cache_data
def get_unique_genres(df):

    cleaned_genres = df['genres'].str.replace('[', '').str.replace(']', '').str.replace("'", "")
    list_of_list = [x.split(',') for x in cleaned_genres.unique() if x != '']
    flat_list = [item.strip() for sublist in list_of_list for item in sublist]
    unique_genres = sorted(list(set(flat_list)))
    return unique_genres

def search_sth_fast(search_term: str) -> List[str]:
    start_time = time()
    df = load_dataset()
    if not search_term:
        return []
    
    search_term = search_term.strip()

    result = df.loc[df['original_title'].str.contains(search_term, case=False)]\
        .sort_values(by='vote_average', ascending=False)[['original_title', 'release_date_year']].values

    formatted_result = [f"{name_tuple[0]} ({name_tuple[1]})" for name_tuple in result]
    print(time()-start_time)
    return formatted_result



#################################
#### application starts here ####
#################################


df = load_dataset()

unique_genres = get_unique_genres(df)

selected_genre = 'Pas de genre'
unique_genres.insert(0, 'Pas de genre')
with st.sidebar:
    selected_genre= st.selectbox(
        "Entrez le nom d'un genre",
        unique_genres,
        key="search_genres",
    )
    if selected_genre:
        st.info(f"Filtre par {selected_genre}")





def get_recommandations(df, selected_genre, selected_movie):
    selected_movie = selected_movie[:-7].strip()
    from sklearn.neighbors import NearestNeighbors
    if selected_genre != 'Pas de genre':
        rec_df = df.loc[df['genres'].str.contains(selected_genre)]
    else:
        rec_df = df
    
    movie_without = rec_df.loc[rec_df['original_title'] != selected_movie]
    X = movie_without.select_dtypes('number')
    model = NearestNeighbors(n_neighbors=3)
    model.fit(X)

    stats = rec_df.loc[df['original_title'] == selected_movie, X.columns]
    result = model.kneighbors(stats)
    distance = result[0][0]
    index = result[1][0]
    recommendation = movie_without.iloc[index]
    
    # samples = rec_df.sample(3)
    image_links = recommendation['poster_path'].values
    movie_names = recommendation['original_title'].values
    imdb_ids = recommendation['imdb_id'].values

    return image_links, movie_names, imdb_ids
        

selected_movie = st_searchbox(
    search_sth_fast,
    default=None,
    label="Entrez le nom d'un film",
    clear_on_submit=True,
    key="search_sth_fast"
)





if selected_movie:
    st.title(f'Recommandations liées à {selected_movie}')

    image_links, movie_names, imdb_ids = get_recommandations(df, selected_genre, selected_movie)

    c0, c1, c2 = st.columns(3)

    with c0:
        full_link_0 = 'https://image.tmdb.org/t/p/w500/'+image_links[0]
        st.image(full_link_0, use_column_width='auto')
        imdb_link_0 = 'https://www.imdb.com/title/'+imdb_ids[0]
        st.header(f"[{movie_names[0]}]({imdb_link_0})")

    with c1:
        full_link_1 = 'https://image.tmdb.org/t/p/w500/'+image_links[1]
        st.image(full_link_1, use_column_width='auto')
        imdb_link_1 = 'https://www.imdb.com/title/'+imdb_ids[1]
        st.header(f"[{movie_names[1]}]({imdb_link_1})")

    with c2:
        full_link_2 = 'https://image.tmdb.org/t/p/w500/'+image_links[2]
        st.image(full_link_2, use_column_width='auto')
        imdb_link_2 = 'https://www.imdb.com/title/'+imdb_ids[2]
        st.header(f"[{movie_names[2]}]({imdb_link_2})")
