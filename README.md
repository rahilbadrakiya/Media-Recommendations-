# CineMate - Intelligent Movie Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/rahilbadrakiya/Media-Recommendations-)

**CineMate** is a full-stack movie recommendation application that provides personalized suggestions using machine learning. It features a modern, Netflix-inspired dark UI and a custom-built K-Nearest Neighbors (KNN) algorithm to find movies similar to your favorites.

The application allows users to browse trending content, get recommendations based on specific movies or genres, and maintain a personal profile of liked movies.

## Key Features

### 🧠 Intelligent Recommendation Engine
*   **KNN Algorithm**: Uses a custom K-Nearest Neighbors implementation (Euclidean distance) to identify movies with similar feature vectors (genres, ratings).
*   **Dual-Mode Discovery**:
    *   **Movie-Based**: Select a movie you love, and the engine finds geographically close vectors in the feature space.
    *   **Genre-Based**: Select multiple genres and a minimum IMDb rating to generate a custom playlist.
*   **Dynamic "Load More"**: content is fetched in batches (default 60), allowing endless scrolling without page reloads.

### 🎨 Modern User Experience
*   **Netflix-Style Dark Theme**: Custom CSS implementation with glassmorphism, hover scaling effects, and a cinematographic color palette (Black/Red/White).
*   **Live Poster Fetching**: Scrapes IMDb meta tags in real-time to display the most up-to-date high-quality movie posters.
*   **Responsive Grid Layout**: Adaptive movie cards that look great on desktop and larger screens.

### 👤 User Accounts & Profile
*   **Secure Authentication**: Registration and Login system with SHA-256 password hashing.
*   **Favorites System**: Interactive "Heart" icon on every movie card.
*   **Personalized Profile**: "Liked" movies are saved to a persistent SQLite database and displayed in the user's profile.
*   **Guest Handling**: Unauthenticated users are gracefully redirected to login when attempting to interact with features.

---

## Technical Architecture

### Tech Stack
*   **Frontend**: Streamlit (Python-based web framework)
*   **Backend Logic**: Python 3.9+
*   **Machine Learning**: Scikit-learn, NumPy, Custom KNN Class
*   **Data Processing**: Pandas
*   **Database**: SQLite (UserInfo table with JSON-serialized lists for likes/genres)
*   **Web Scraping**: BeautifulSoup4, Requests (for fetching poster images)

### Project Structure
```
├── App.py                # Main application controller & UI rendering
├── Classifier.py         # Custom KNN Algorithm implementation
├── Data/
│   ├── movie_data.json   # Pre-processed feature vectors for the model
│   └── movie_titles.json # Metadata (ID, Title, IMDb Link)
├── assets/
│   └── custom.css        # Custom CSS for styling components
├── users.db              # Local database for user credentials & history
└── requirements.txt      # Python dependencies
```

---

## Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/rahilbadrakiya/Media-Recommendations-.git
    cd Media-Recommendations-
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    ```bash
    streamlit run App.py
    ```

The app will launch in your default browser at `http://localhost:8501`.

---

## Future Roadmap

*   **Hybrid Filtering**: Implement Content-Based Filtering to solve the "cold start" problem for new movies.
*   **API Integration**: Migrate from scraping to the TMDB API for faster image loading, trailers, and cast details.
*   **Social Features**: Allow users to share their "Liked" lists with friends.
*   **Cloud Deployment**: Migrate the SQLite database to PostgreSQL for persistent hosting on platforms like Streamlit Cloud or Render.

---

## Contributing

Contributions are welcome!

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
