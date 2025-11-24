import streamlit as st
from PIL import Image
import json
import sqlite3
import random
import hashlib
from Classifier import KNearestNeighbours
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import os

# Cacheable functions
@st.cache_data
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_pickle(path):
    return pd.read_pickle(path)

@st.cache_data
def get_movie_poster(imdb_link):
    try:
        url_data = requests.get(imdb_link, headers={'User-Agent': 'Mozilla/5.0'}).text
        s_data = BeautifulSoup(url_data, 'html.parser')
        poster = s_data.find("meta", property="og:image")
        return poster['content'] if poster and poster.has_attr('content') else None
    except:
        return None

@st.cache_data
def get_movie_recommendations(test_point, k):
    target = [0 for _ in movie_titles]
    model = KNearestNeighbours(data, target, test_point, k=k)
    model.fit()
    return [[movie_titles[i][0], movie_titles[i][2], data[i][-1]] for i in model.indices]

@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect("users.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    return conn, cursor

# Load data
movie_data_path = os.path.join("Data", "movie_data.json")
movie_titles_path = os.path.join("Data", "movie_titles.json")
book_model_path = "book_model"

data = load_json(movie_data_path)
movie_titles = load_json(movie_titles_path)

popular_df = load_pickle(os.path.join(book_model_path, "popular.pkl"))
pt = load_pickle(os.path.join(book_model_path, "pt.pkl"))
books = load_pickle(os.path.join(book_model_path, "books.pkl"))
similarity_scores = load_pickle(os.path.join(book_model_path, "similarity_scores.pkl"))

conn, cursor = get_db_connection()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register():
    st.subheader("Register")
    username = st.text_input("Username", key="reg_username")
    password = st.text_input("Password", type="password", key="reg_password")
    if st.button("Register", key="register_btn"):
        hashed = hash_password(password)
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            conn.commit()
            st.success("Registered successfully! Please login.")
        except sqlite3.IntegrityError:
            st.error("Username already exists!")
def login():
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", key="login_btn"):
        hashed = hash_password(password)
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed))
        user = cursor.fetchone()
        if user:
            st.session_state["logged_in"] = True
            st.success("Logged in successfully!")
            st.rerun()  # Replaced st.experimental_rerun() with st.rerun()
        else:
            st.error("Invalid credentials.")

def logout():
    st.session_state["logged_in"] = False
    st.session_state.pop("user", None)
    st.success("Logged out successfully!")
    st.rerun()  # Replaced st.experimental_rerun() with st.rerun()


def movie_poster_fetcher(imdb_link):
    poster_url = get_movie_poster(imdb_link)
    if poster_url:
        st.image(poster_url, use_container_width=False)
    else:
        st.warning("Poster not available.")

def recommendation_page():
    st.subheader("Random Movie Recommendations")
    random_movies = random.sample(movie_titles, 15)
    for movie in random_movies:
        st.markdown(f"### {movie[0]} ({data[movie_titles.index(movie)][-1]} ⭐)")
        movie_poster_fetcher(movie[2])

def book_recommendation_page():
    st.subheader("📚 Book Recommendation")
    book_list = list(pt.index)
    selected_book = st.selectbox("Select a book:", ['--Select--'] + book_list)
    if selected_book != '--Select--':
        index = np.where(pt.index == selected_book)[0][0]
        similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]
        st.success("Here are some books you might like:")
        for i, score in similar_items:
            book_title = pt.index[i]
            book_data = books[books['Book-Title'].isin([book_title])].drop_duplicates('Book-Title')
            title = book_data['Book-Title'].values[0]
            author = book_data['Book-Author'].values[0]
            img_url = book_data['Image-URL-M'].values[0]
            st.image(img_url, width=150)
            st.markdown(f"**Title:** {title}")
            st.markdown(f"**Author:** {author}")
            st.markdown("---")

def top_books_page():
    st.subheader("Top 50 Books")
    top_books = popular_df.head(50)
    if top_books.empty:
        st.warning("No books data available.")
    else:
        for idx, row in top_books.iterrows():
            st.markdown(f"**{row['Book-Title']}** by {row['Book-Author']}")
            st.image(row["Image-URL-M"], width=150)
            if "avg_rating" in row:
                st.markdown(f"Average Rating: {row['avg_rating']}")
            st.markdown("---")

def home_page():
    try:
        img_path = os.path.join("meta", "logo.jpg")
        img1 = Image.open(img_path).resize((250, 250))
        st.image(img1, use_container_width=False)
    except Exception:
        pass
    st.title("🎮 Movie Recommender System")
    st.markdown("<h4 style='color: #d73b5c;'>* Based on IMDb 5000 Movie Dataset</h4>", unsafe_allow_html=True)
    genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama',
              'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music', 'Musical',
              'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    cat_op = st.selectbox('Select Recommendation Type', ['--Select--', 'Movie based', 'Genre based'])
    if cat_op == '--Select--':
        st.warning("Please select a recommendation type!")
    elif cat_op == 'Movie based':
        movies = [title[0] for title in movie_titles]
        select_movie = st.selectbox('Select movie:', ['--Select--'] + movies)
        fetch_poster = st.radio("Fetch Movie Poster?", ('Yes', 'No'))
        if select_movie != '--Select--':
            no_of_reco = st.slider('Number of recommendations:', 5, 20, 5)
            test_points = data[movies.index(select_movie)]
            recommendations = get_movie_recommendations(test_points, no_of_reco + 1)[1:]
            st.success('Here are some recommended movies:')
            for idx, (movie, link, rating) in enumerate(recommendations, 1):
                st.markdown(f"({idx}) [{movie}]({link})")
                if fetch_poster == 'Yes':
                    movie_poster_fetcher(link)
                st.markdown(f'IMDb Rating: {rating} ⭐')
    elif cat_op == 'Genre based':
        sel_gen = st.multiselect('Select Genres:', genres)
        fetch_poster = st.radio("Fetch Movie Poster?", ('Yes', 'No'))
        if sel_gen:
            imdb_score = st.slider('Choose IMDb score:', 1, 10, 8)
            no_of_reco = st.number_input('Number of movies:', 5, 20, step=1)
            test_point = [1 if genre in sel_gen else 0 for genre in genres] + [imdb_score]
            recommendations = get_movie_recommendations(test_point, no_of_reco)
            st.success('Here are some recommended movies:')
            for idx, (movie, link, rating) in enumerate(recommendations, 1):
                st.markdown(f"({idx}) [{movie}]({link})")
                if fetch_poster == 'Yes':
                    movie_poster_fetcher(link)
                st.markdown(f'IMDb Rating: {rating} ⭐')

def run():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # Sidebar user profile and logout
    with st.sidebar:
        if st.session_state["logged_in"]:
            st.markdown("---")
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image("https://cdn-icons-png.flaticon.com/512/149/149071.png", width=30)
            with col2:
                st.markdown("**Logged in**")
            if st.button("Logout"):
                logout()
            st.markdown("---")

    if st.session_state["logged_in"]:
        nav_options = ["Home", "Recommendation", "Books", "Top 50 Books"]
    else:
        nav_options = ["Login", "Register", "Recommendation", "Top 50 Books"]

    page = st.sidebar.radio("Navigation", nav_options)
    if page == "Login":
        login()
    elif page == "Register":
        register()
    elif page == "Recommendation":
        recommendation_page()
    elif page == "Books":
        if st.session_state["logged_in"]:
            book_recommendation_page()
        else:
            st.warning("Please log in to access the Books page.")
    elif page == "Top 50 Books":
        top_books_page()
    elif page == "Home":
        if st.session_state["logged_in"]:
            home_page()
        else:
            st.warning("Please log in to access the home page.")

run()