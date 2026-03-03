# 🎬 CineMate Ultra — AI-Powered Movie Recommendation Engine

> A Netflix-inspired movie discovery platform combining a **Hybrid Recommendation Algorithm**, **AI Chatbot (Gemini)**, and **real-time TMDB data** for both Hollywood and Indian cinema.

![Tech Stack](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat&logo=vite&logoColor=white)
![TMDB](https://img.shields.io/badge/TMDB-01B4E4?style=flat&logo=themoviedatabase&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_AI-4285F4?style=flat&logo=google&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=flat&logo=firebase&logoColor=black)

---

## ✨ Features

### 🧠 Netflix-Level Hybrid Recommendation Engine
Combines 5 signals into one score:
```
Final Score = 0.30 × User Preference
            + 0.20 × Trending Score
            + 0.20 × Similarity Score
            + 0.15 × Recency Score
            + 0.15 × Popularity Score
```

### 🤖 AI Chatbot (Gemini-Powered)
- Natural language prompts: *"dark thriller like Se7en but modern"*
- Understands mood, vibe, genre, era, and runtime preferences
- Renders movie cards directly inside the chat panel
- Falls back to keyword-based NLP if no API key is set

### 🎬 Smart Recommendation Sections
| Section | Description |
|---|---|
| 🧠 AI Picks For You | Hybrid score — personalized for logged-in users |
| 🔥 Just Released For You | Last 60 days, filtered by your genre preferences |
| 🎞️ Old Classics Like Trending | Timeless movies matching current trending vibes |
| 🍿 Because You Loved | Multi-movie BYW using all liked + rated movies |
| 📅 Contextual / Seasonal | Festival/seasonal picks based on today's date |
| 🎭 Mood Selector | 8 moods with persistent last-selection |
| ⏱️ Time Filter | Any / Under 90 min / Under 2 hrs |

### 🇮🇳 Indian Cinema Support
- Bollywood, Tollywood, Kollywood, Mollywood, Sandalwood
- All recommendation features work for Indian industry too
- Separate trending, now-playing, upcoming, and top-rated

### 👤 User Features
- Firebase Auth (Email + Google Sign-In)
- Like, Watchlist, and Star Rating (1–5)
- Watch history tracking (powers hybrid recommendations)
- Personalized genre preferences at registration

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Vanilla JS, Vite, CSS3 |
| **Backend** | Python, FastAPI |
| **Recommendation ML** | Custom KNN classifier |
| **Movie Data** | TMDB API |
| **AI Chatbot** | Google Gemini 2.0 Flash |
| **Auth** | Firebase Authentication |
| **Database** | SQLite (local user data) |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- TMDB API key (free at [themoviedb.org](https://www.themoviedb.org/settings/api))
- Gemini API key (free at [aistudio.google.com](https://aistudio.google.com/app/apikey))

### 1. Clone the repo
```bash
git clone https://github.com/rahilbadrakiya/Media-Recommendations-.git
cd Media-Recommendations-
```

### 2. Setup environment variables
```bash
cp .env.example .env
# Edit .env and add your TMDB_API_KEY and GEMINI_API_KEY
```

### 3. Install backend dependencies
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r backend/requirements.txt
```

### 4. Install frontend dependencies
```bash
cd frontend
npm install
cd ..
```

### 5. Run the app
```bash
# Windows (double-click or run):
start.bat

# Manual:
# Terminal 1 — Backend
cd backend && python main.py

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Open **http://localhost:5173** in your browser.

---

## 📁 Project Structure

```
CineMate/
├── backend/
│   ├── main.py              # FastAPI app — all API endpoints
│   ├── Classifier.py        # KNN recommendation model
│   ├── tmdb_utils.py        # TMDB API client
│   ├── requirements.txt     # Python dependencies
│   └── Data/
│       ├── movie_data.json  # Feature vectors for KNN
│       └── movie_titles.json
│
├── frontend/
│   ├── index.html           # Main HTML + chatbot HTML
│   ├── main.js              # App logic, routing, all UI
│   ├── style.css            # Complete design system
│   ├── firebase.js          # Firebase auth
│   └── vite.config.js
│
├── .env.example             # Template for env variables
├── .gitignore
├── start.bat                # One-click start (Windows)
├── Movie_Data_Processing.ipynb  # Data preprocessing notebook
└── README.md
```

---

## 🔑 API Endpoints

### Recommendations
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/recommendations/hybrid` | Netflix hybrid score engine |
| GET | `/api/recommendations/for-you` | Multi-movie BYW |
| GET | `/api/recommendations/just-released` | Last 60 days + genre filter |
| GET | `/api/recommendations/classics-like-trending` | Old classics matching trending |
| POST | `/api/recommendations/movie` | KNN movie-based |
| POST | `/api/recommendations/genre` | KNN genre-based |
| POST | `/api/recommendations/mood` | KNN mood-based |

### AI Chatbot
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/chat` | Gemini AI natural language recommendations |

### Movies (TMDB)
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/movies/trending` | Hollywood trending |
| GET | `/api/movies/now-playing` | Now in theatres |
| GET | `/api/movies/upcoming` | Coming soon |
| GET | `/api/movies/contextual` | Seasonal/festival picks |
| GET | `/api/movies/indian-trending` | Indian cinema trending |
| GET | `/api/movies/{id}/details` | Trailers, cast, similar |
| GET | `/api/movies/search` | Search movies |

### User
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/register` | Create account |
| POST | `/api/auth/login` | Email/password login |
| POST | `/api/auth/firebase` | Firebase auth sync |
| POST | `/api/user/likes` | Toggle like |
| POST | `/api/user/watchlist` | Toggle watchlist |
| POST | `/api/user/rate` | Rate a movie (1–5) |
| POST | `/api/user/history` | Track watch history |

---

## 📸 Screenshots

> Hero section, mood selector, AI chatbot, and movie cards

_(Add your screenshots to the `Screenshots/` folder and link them here)_

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🙏 Credits

- Movie data: [TMDB API](https://www.themoviedb.org/)
- AI: [Google Gemini](https://deepmind.google/technologies/gemini/)
- Auth: [Firebase](https://firebase.google.com/)
- Design inspired by Netflix & modern streaming UIs
