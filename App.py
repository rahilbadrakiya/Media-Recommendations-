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
@st.cache_data(show_spinner=False)
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_pickle(path):
    return pd.read_pickle(path)

@st.cache_data(show_spinner=False)
def get_movie_poster(imdb_link):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        response = requests.get(imdb_link, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None
            
        s_data = BeautifulSoup(response.text, 'html.parser')
        poster = s_data.find("meta", property="og:image")
        return poster['content'] if poster and poster.has_attr('content') else None
    except Exception as e:
        print(f"Error fetching poster for {imdb_link}: {e}")
        return None

@st.cache_data(show_spinner=False)
def get_movie_recommendations(test_point, k):
    target = [0 for _ in movie_titles]
    model = KNearestNeighbours(data, target, test_point, k=k)
    model.fit()
    return [[movie_titles[i][0], movie_titles[i][2], data[i][-1]] for i in model.indices]

@st.cache_resource
def get_db_connection():
    # Added timeout to prevent "database is locked" errors
    conn = sqlite3.connect("users.db", check_same_thread=False, timeout=30)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            fav_genres TEXT,
            liked_movies TEXT
        )
    """)
    
    # Safe Migrations (ignore if fails due to lock/existence)
    try:
        cursor.execute("SELECT fav_genres FROM users LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN fav_genres TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass # Ignore if locked or already added
            
    try:
        cursor.execute("SELECT liked_movies FROM users LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN liked_movies TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass # Ignore if locked or already added
            
    conn.commit()
    return conn, cursor

def get_user_likes(username):
    cursor.execute("SELECT liked_movies FROM users WHERE username=?", (username,))
    result = cursor.fetchone()
    if result and result[0]:
        return result[0].split(',')
    return []

def toggle_like(username, movie_title):
    current_likes = get_user_likes(username)
    
    if movie_title in current_likes:
        current_likes.remove(movie_title)
    else:
        current_likes.append(movie_title)
    
    new_likes_str = ",".join(current_likes)
    cursor.execute("UPDATE users SET liked_movies=? WHERE username=?", (new_likes_str, username))
    conn.commit()
    return current_likes

# Load data
movie_data_path = os.path.join("Data", "movie_data.json")
movie_titles_path = os.path.join("Data", "movie_titles.json")

data = load_json(movie_data_path)
movie_titles = load_json(movie_titles_path)


conn, cursor = get_db_connection()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register():
    st.markdown("""
        <div class="custom-header">
            <h1>Sign Up</h1>
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("register_form"):
        username = st.text_input("Username", max_chars=50)
        password = st.text_input("Password", type="password", help="Must be at least 6 characters", max_chars=50)
        confirm_password = st.text_input("Confirm Password", type="password", max_chars=50)
        
        genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama',
              'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music', 'Musical',
              'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
        
        selected_genres = st.multiselect("What do you like to watch?", genres)
        
        submit = st.form_submit_button("Create Account")
        
        if submit:
            if not username or not password:
                st.error("Please fill in all required fields.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters long.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            elif len(selected_genres) < 3:
                st.warning("Please select at least 3 favorite genres for better recommendations.")
            else:
                hashed = hash_password(password)
                genres_str = ",".join(selected_genres)
                try:
                    cursor.execute("INSERT INTO users (username, password, fav_genres) VALUES (?, ?, ?)", (username, hashed, genres_str))
                    conn.commit()
                    st.success("Registered successfully! Please login.")
                except sqlite3.IntegrityError:
                    st.error("Username already exists!")

def login():
    st.markdown("""
        <div class="custom-header">
            <h1>Sign In</h1>
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("Username", max_chars=50)
        password = st.text_input("Password", type="password", max_chars=50)
        submit = st.form_submit_button("Sign In")
        
        if submit:
            hashed = hash_password(password)
            cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed))
            user = cursor.fetchone()
            if user:
                st.session_state["logged_in"] = True
                st.session_state["username"] = user[1]
                # user[3] is fav_genres
                st.session_state["user_genres"] = user[3].split(",") if user[3] else []
                st.success("Logged in successfully!")
                set_page("Home")
                st.rerun()
            else:
                st.error("Invalid credentials.")

def logout():
    st.session_state["logged_in"] = False
    st.session_state.pop("user", None)
    st.session_state.pop("user_genres", None)
    st.session_state.pop("username", None)
    st.success("Logged out successfully!")
    set_page("Home") # Go to landing
    st.rerun()


def movie_poster_fetcher(imdb_link):
    poster_url = get_movie_poster(imdb_link)
    if poster_url:
        st.image(poster_url, use_container_width=False)
    else:
        st.warning("Poster not available.")




def load_local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def get_movies_by_genre(genre, limit=12):
    """Fetch top movies for a specific genre based on IMDb rating."""
    genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama',
              'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music', 'Musical',
              'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    
    if genre not in genres:
        return []
    
    genre_idx = genres.index(genre)
    
    # Filter movies that have this genre (1 at the genre index)
    # data[i] structure: [genre_1, genre_2, ..., genre_n, imdb_score]
    # We want indices where data[i][genre_idx] == 1
    
    # Let's create a list of (index, rating)
    genre_movies = []
    for idx, row in enumerate(data):
        if row[genre_idx] == 1:
            genre_movies.append((idx, row[-1])) # idx, rating
            
    # Sort by rating descending
    genre_movies.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 'limit'
    top_indices = [x[0] for x in genre_movies[:limit]]
    
    return [movie_titles[i] for i in top_indices]

def display_movie_row(title, movies, key_suffix):
    st.markdown(f"### {title}")
    
    # Session state for this row
    sess_key = f"count_{key_suffix}"
    if sess_key not in st.session_state:
        st.session_state[sess_key] = 60
        
    current_count = st.session_state[sess_key]
    movies_to_show = movies[:current_count]
    
    # We want to display in rows of 6
    cols_per_row = 6
    # Chunk the movies list
    for i in range(0, len(movies_to_show), cols_per_row):
        chunk = movies_to_show[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        
        for idx, movie in enumerate(chunk):
            with cols[idx]:
                with st.container():
                     link = movie[2]
                     title_text = movie[0]
                     
                     # Check if poster fetch is enabled globally or just use placeholders for speed/api limits
                     # For the home page with MANY movies, live fetching might be slow or blocked. 
                     # Let's use a mix or just fetch for now (since user wanted "nice aesthetic")
                     # ideally we'd cache this heavily.
                     poster = get_movie_poster(link)
                     poster_src = poster if poster else "https://via.placeholder.com/200x300?text=No+Poster"
                     
                     st.markdown(f"""
                        <div class="movie-card">
                            <a href="{link}" target="_blank" style="text-decoration: none; color: inherit;">
                                <img src="{poster_src}" style="width: 100%; border-radius: 5px; transition: transform 0.3s; object-fit: cover; aspect-ratio: 2/3;">
                                <div class="movie-title">{title_text}</div>
                            </a>
                        </div>
                     """, unsafe_allow_html=True)
                     
    if current_count < len(movies):
        if st.button("Load More", key=f"btn_{key_suffix}"):
            st.session_state[sess_key] += 30
            st.rerun()
                     
def home_page():
    # --- FEATURED MOVIE HERO SECTION (Cached) ---
    if "home_hero_movie" not in st.session_state:
        try:
            # Pick a random movie from top 100 for better covers
            featured_idx = random.choice([i for i, r in enumerate(data) if r[-1] > 8.5])
            featured_movie = movie_titles[featured_idx]
            f_title = featured_movie[0]
            f_link = featured_movie[2]
            
            poster_url = get_movie_poster(f_link)
            bg_image = poster_url if poster_url else "https://images.unsplash.com/photo-1536440136628-849c177e76a1?ixlib=rb-1.2.1&auto=format&fit=crop&w=1925&q=80"
            
            st.session_state["home_hero_movie"] = {
                "title": f_title,
                "bg_image": bg_image
            }
        except:
             st.session_state["home_hero_movie"] = {
                "title": "The Dark Knight",
                "bg_image": "https://image.tmdb.org/t/p/original/qJ2tW6WMUDux911r6m7haRef0WH.jpg"
            }
            
    hero_data = st.session_state["home_hero_movie"]
    
    # HTML for Hero
    st.markdown(f"""
        <div class="hero-container" style="background-image: url('{hero_data['bg_image']}');">
            <div class="hero-overlay">
                <div class="hero-title">{hero_data['title']}</div>
                <div class="hero-desc">
                    Watch the highest rated movies selected just for you. Experience cinema like never before with our curated playlists and personalized engine.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 1. Personalized Recommendations (if logged in and has genres)
    if st.session_state.get("logged_in") and st.session_state.get("user_genres"):
        st.markdown("### 🌟 For You")
        user_genres = st.session_state["user_genres"]
        for genre in user_genres[:3]:
            titles = {
                "Action": "High-Octane Action", "Comedy": "Laugh Out Loud", 
                "Horror": "Spine-Chilling Picks", "Romance": "Love & Romance",
                "Sci-Fi": "Sci-Fi Adventures", "Drama": "Critical Acclaim"
            }
            title = titles.get(genre, f"Top {genre} Movies")
            # Fetch more movies (e.g. 120) so we can page through them
            movies = get_movies_by_genre(genre, 120) 
            display_movie_row(title, movies, f"home_genre_{genre}")
            
    # 2. Trending Data (Cached)
    if "home_trending_movies" not in st.session_state:
        high_rated_indices = [i for i, row in enumerate(data) if row[-1] > 8.0]
        # Fetch more trending movies
        trending_indices = random.sample(high_rated_indices, min(120, len(high_rated_indices)))
        trending_movies = [movie_titles[i] for i in trending_indices]
        st.session_state["home_trending_movies"] = trending_movies
        
    display_movie_row("Trending Now", st.session_state["home_trending_movies"], "home_trending")
    
    # Genre Rows (Static)
    display_movie_row("Action Hits", get_movies_by_genre("Action", 120), "home_action")
    display_movie_row("Comedy Favorites", get_movies_by_genre("Comedy", 120), "home_comedy")


def recommendation_page():
    st.markdown("""
        <div class="custom-header">
            <h1>Movie Recommendations</h1>
        </div>
    """, unsafe_allow_html=True)

    genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama',
              'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music', 'Musical',
              'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    
    cat_op = st.selectbox('Select Recommendation Type', ['--Select--', 'Movie based', 'Genre based'])

    if cat_op == 'Movie based':
        movies = [title[0] for title in movie_titles]
        select_movie = st.selectbox('Select movie:', ['--Select--'] + movies)
        
        if select_movie != '--Select--':
            # Session State for Load More
            if 'last_movie' not in st.session_state or st.session_state['last_movie'] != select_movie:
                st.session_state['last_movie'] = select_movie
                st.session_state['movie_reco_count'] = 60
            
            current_count = st.session_state['movie_reco_count']
            
            # Fetch recommendations (fetch current count)
            try:
                movie_idx = movies.index(select_movie)
                test_points = data[movie_idx]
                # We fetch all needed + 1 (itself)
                recommendations = get_movie_recommendations(test_points, current_count + 1)[1:]
                
                st.markdown("### More Like This")
                cols = st.columns(6)
                
                for idx, (movie, link, rating) in enumerate(recommendations):
                     with cols[idx % 6]:
                        poster = get_movie_poster(link)
                        is_liked = movie in user_likes
                        st.markdown(get_movie_card_html(movie, link, rating, poster, is_liked), unsafe_allow_html=True)
                        
                # Load More Button
                if st.button("Load More", key="load_more_movie"):
                    st.session_state['movie_reco_count'] += 30
                    st.rerun()
                    
            except ValueError:
                st.error("Movie not found in database.")

    elif cat_op == 'Genre based':
        sel_gen = st.multiselect('Select Genres:', genres)
        
        if sel_gen:
            # Session State for Load More
            # Convert list to tuple for hashable comparison
            gen_tuple = tuple(sorted(sel_gen))
            if 'last_genres' not in st.session_state or st.session_state['last_genres'] != gen_tuple:
                st.session_state['last_genres'] = gen_tuple
                st.session_state['genre_reco_count'] = 60
                
            current_count = st.session_state['genre_reco_count']
        
            imdb_score = st.slider('Minimum IMDb score:', 1, 10, 8)
            # Removed number_input
            
            test_point = [1 if genre in sel_gen else 0 for genre in genres] + [imdb_score]
            recommendations = get_movie_recommendations(test_point, current_count)
            
            st.markdown("### Top Picks for You")
            cols = st.columns(6)
            
            for idx, (movie, link, rating) in enumerate(recommendations):
                with cols[idx % 6]:
                    poster = get_movie_poster(link)
                    is_liked = movie in user_likes
                    st.markdown(get_movie_card_html(movie, link, rating, poster, is_liked), unsafe_allow_html=True)

            # Load More Button
            if st.button("Load More", key="load_more_genre"):
                st.session_state['genre_reco_count'] += 30
                st.rerun()


def set_page(page_name):
    st.session_state["page"] = page_name

def footer():
    st.markdown("""
        <div style="text-align: center; margin-top: 50px; padding: 20px; color: #666; font-size: 0.8rem;">
            <p>CINEMATE © 2026</p>
            <p>Designed for Movie & Book Lovers</p>
        </div>
    """, unsafe_allow_html=True)

def get_movie_card_html(movie, link, rating, poster, is_liked=False):
    heart_color = "❤️" if is_liked else "🤍"
    like_link = f"?toggle_like={movie}"
    
    poster_src = poster if poster else "https://via.placeholder.com/200x300?text=No+Poster"
    
    return f"""
    <div class="movie-card">
        <div class="heart-icon" title="Like this movie">
            <a href="{like_link}" target="_self">{heart_color}</a>
        </div>
        <a href="{link}" target="_blank" style="text-decoration: none; color: inherit;">
            <img src="{poster_src}" style="width: 100%; border-radius: 5px; object-fit: cover; aspect-ratio: 2/3;">
            <div class="movie-title">{movie}</div>
        </a>
        <div class="movie-rating">{rating}</div>
    </div>
    """

def handle_likes():
    # Check for query params
    # Streamlit >= 1.30 uses st.query_params
    query_params = st.query_params
    
    if "toggle_like" in query_params:
        movie_to_toggle = query_params["toggle_like"]
        
        # If not logged in, redirect to Login
        if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
            st.warning("Please sign in to like movies.")
            set_page("Login")
            st.query_params.clear()
            st.rerun()
            return

        toggle_like(st.session_state["username"], movie_to_toggle)
        
        # Clear the param
        st.query_params.clear()
        st.rerun()

def navbar():
    st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
    
    with st.container():
        # Simplified Layout: Logo | Home | Browse (if logged in) | Spacer | Profile/Auth
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 2, 2.5]) # Widened col5
        
        with col1:
             st.markdown('<div class="nav-logo">CINEMATE</div>', unsafe_allow_html=True)
        
        with col2:
            if st.button("Home"):
                set_page("Home")
        
        with col3:
            if st.session_state.get("logged_in"):
                if st.button("Browse"):
                    set_page("Recommendation")
        
        # col4 is spacer
        
        # User Action Section
        with col5:
            if st.session_state.get("logged_in"):
                # Profile / Logout - Side by Side
                c_profile, c_logout = st.columns([2, 1])
                with c_profile:
                     # Button to go to Profile
                     if st.button(f"👤 {st.session_state.get('username', 'User')}"):
                        set_page("Profile")
                with c_logout:
                    if st.button("Logout"):
                        logout()
            else:
                # Login / Join - Left aligned with margin
                col_a, col_b, _ = st.columns([1, 1, 3])
                with col_a:
                    if st.button("Login"):
                        set_page("Login")
                with col_b:
                    if st.button("Join"): # Short for Join/Sign Up
                        set_page("Register")
        
        st.markdown("---")

def run():
    st.set_page_config(page_title="CineMate", page_icon="🎬", layout="wide") 
    
    # Load Custom CSS
    css_path = os.path.join("assets", "custom.css")
    if os.path.exists(css_path):
        load_local_css(css_path)
    
    # Session State Initialization
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    
    # Handle Likes (Query Params)
    handle_likes()
    
    # Navbar
    navbar()
    
    # Page Routing
    page = st.session_state.get("page", "Home")
    
    if page == "Home":
        home_page()
            
    elif page == "Recommendation":
        recommendation_page()
        
    elif page == "Profile":
        if not st.session_state["logged_in"]:
            set_page("Login")
            st.rerun()

        st.markdown("### Your Favorites ❤️")
        user_likes = get_user_likes(st.session_state["username"])
        
        if not user_likes:
            st.info("You haven't liked any movies yet. Go explore and tap the heart icon!")
        else:
            liked_details = []
            all_movies_list = [title[0] for title in movie_titles]
            
            for liked_movie in user_likes:
                if liked_movie in all_movies_list:
                    idx = all_movies_list.index(liked_movie)
                    link = movie_titles[idx][2]
                    rating = data[idx][-1]
                    liked_details.append((liked_movie, link, rating))
            
            if liked_details:
                cols = st.columns(6)
                for idx, (movie, link, rating) in enumerate(liked_details):
                    with cols[idx % 6]:
                        poster = get_movie_poster(link)
                        st.markdown(get_movie_card_html(movie, link, rating, poster, is_liked=True), unsafe_allow_html=True)
            else:
                 st.warning("Could not load details for your liked movies.")

    elif page == "Register":
        register()

    elif page == "Login":
        login()

    footer()

if __name__ == '__main__':
    run()