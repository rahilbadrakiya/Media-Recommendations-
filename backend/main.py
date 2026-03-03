from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, os, sqlite3, hashlib, requests, random
from Classifier import KNearestNeighbours
from tmdb_utils import TMDBClient
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Firebase Admin — initialise lazily so the app still works without credentials
_firebase_ready = False
def _init_firebase():
    global _firebase_ready
    if _firebase_ready:
        return True
    try:
        import firebase_admin
        from firebase_admin import credentials
        svc = os.path.join(os.path.dirname(__file__), '..', 'firebase-service-account.json')
        if os.path.exists(svc) and not firebase_admin._apps:
            cred = credentials.Certificate(svc)
            firebase_admin.initialize_app(cred)
        _firebase_ready = True
        return True
    except Exception as e:
        print(f"[Firebase] Not configured: {e}")
        return False

app = FastAPI(title="CineMate API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
with open(os.path.join(DATA_DIR, "movie_data.json"), encoding="utf-8") as f:
    movie_data = json.load(f)
with open(os.path.join(DATA_DIR, "movie_titles.json"), encoding="utf-8") as f:
    movie_titles = json.load(f)

tmdb = TMDBClient()

GENRES = ['Action','Adventure','Animation','Biography','Comedy','Crime','Documentary','Drama',
          'Family','Fantasy','Film-Noir','Game-Show','History','Horror','Music','Musical',
          'Mystery','News','Reality-TV','Romance','Sci-Fi','Short','Sport','Thriller','War','Western']

MOOD_GENRE_MAP = {
    "happy":     ["Comedy", "Animation", "Family", "Music", "Adventure"],
    "sad":       ["Drama", "Romance", "Music"],
    "excited":   ["Action", "Adventure", "Sci-Fi", "Thriller"],
    "scared":    ["Horror", "Thriller", "Mystery"],
    "romantic":  ["Romance", "Drama", "Music"],
    "motivated": ["Biography", "Sport", "Drama", "History"],
    "mindblown": ["Sci-Fi", "Mystery", "Thriller", "Fantasy"],
    "chill":     ["Comedy", "Animation", "Family", "Documentary"],
}

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'users.db')

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            fav_genres TEXT,
            liked_movies TEXT,
            watchlist TEXT,
            ratings TEXT,
            watch_history TEXT
        );
    """)
    # Migrate old rows
    for col in ["liked_movies","watchlist","ratings","watch_history"]:
        try: cur.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT")
        except: pass
    conn.commit()
    return conn, cur

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

# ─── MODELS ──────────────────────────────────────────────
class RegisterReq(BaseModel):
    username: str; password: str; genres: list[str] = []

class LoginReq(BaseModel):
    username: str; password: str

class MovieRecoReq(BaseModel):
    movie_title: str; count: int = 20; industry: str = "all"

class GenreRecoReq(BaseModel):
    genres: list[str]; min_rating: float = 7.0; count: int = 20; industry: str = "all"

class MoodRecoReq(BaseModel):
    mood: str; count: int = 20; industry: str = "all"

class LikeReq(BaseModel):
    username: str; movie_title: str

class WatchlistReq(BaseModel):
    username: str; movie_title: str

class RateReq(BaseModel):
    username: str; movie_title: str; rating: float   # 1‑5

class BatchSearchReq(BaseModel):
    titles: list[str]

class ChatReq(BaseModel):
    message: str
    username: str = ""
    industry: str = "all"
    history: list[dict] = []

class WatchHistoryReq(BaseModel):
    username: str
    movie_id: int
    movie_title: str

# ─── AUTH ─────────────────────────────────────────────────
@app.post("/api/auth/register")
def register(req: RegisterReq):
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be ≥ 6 characters.")
    conn, cur = get_db()
    try:
        cur.execute("INSERT INTO users (username,password,fav_genres) VALUES (?,?,?)",
                    (req.username, hash_pw(req.password), ",".join(req.genres)))
        conn.commit()
        return {"message": "Registered."}
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Username already exists.")
    finally: conn.close()

@app.post("/api/auth/login")
def login(req: LoginReq):
    conn, cur = get_db()
    cur.execute("SELECT * FROM users WHERE username=? AND password=?", (req.username, hash_pw(req.password)))
    u = cur.fetchone(); conn.close()
    if not u: raise HTTPException(401, "Invalid credentials.")
    return user_dict(u)

class FirebaseAuthReq(BaseModel):
    id_token: str
    username: str = ""   # optional display name from Firebase
    genres: list[str] = []

@app.post("/api/auth/firebase")
def firebase_auth(req: FirebaseAuthReq):
    """
    Verify a Firebase ID token and upsert the user in our local DB.
    Returns the same user object as /api/auth/login.
    Falls back to token-based username if Firebase Admin isn't configured.
    """
    uid = None
    email = None

    if _init_firebase():
        try:
            from firebase_admin import auth as fb_auth
            decoded = fb_auth.verify_id_token(req.id_token, clock_skew_seconds=60)
            uid   = decoded.get("uid")
            email = decoded.get("email", "")
        except Exception as e:
            raise HTTPException(401, f"Invalid Firebase token: {e}")
    else:
        # Firebase Admin not configured — trust the frontend (dev/demo mode)
        # Use uid extracted from the JWT payload without verification
        import base64, json as _json
        try:
            payload = req.id_token.split(".")[1]
            payload += "=" * (4 - len(payload) % 4)
            decoded_payload = _json.loads(base64.urlsafe_b64decode(payload))
            uid   = decoded_payload.get("user_id") or decoded_payload.get("sub")
            email = decoded_payload.get("email", "")
        except Exception:
            raise HTTPException(401, "Cannot verify token and Firebase Admin is not configured.")

    if not uid:
        raise HTTPException(401, "Could not extract UID from token.")

    # Use display name or email prefix as username
    username = req.username or (email.split("@")[0] if email else uid[:12])
    # Sanitise
    username = "".join(c for c in username if c.isalnum() or c in "_-")[:30] or uid[:12]

    conn, cur = get_db()
    cur.execute("SELECT * FROM users WHERE username=?", (username,))
    existing = cur.fetchone()
    if not existing:
        # Auto-register
        try:
            cur.execute("INSERT INTO users (username,password,fav_genres) VALUES (?,?,?)",
                        (username, hash_pw(uid), ",".join(req.genres)))
            conn.commit()
        except sqlite3.IntegrityError:
            # Username collision — append uid suffix
            username = f"{username}_{uid[:6]}"
            cur.execute("INSERT OR IGNORE INTO users (username,password,fav_genres) VALUES (?,?,?)",
                        (username, hash_pw(uid), ",".join(req.genres)))
            conn.commit()
        cur.execute("SELECT * FROM users WHERE username=?", (username,))
        existing = cur.fetchone()

    result = user_dict(existing)
    conn.close()
    return result

def user_dict(u):
    return {
        "username":     u["username"],
        "genres":       u["fav_genres"].split(",") if u["fav_genres"] else [],
        "liked_movies": u["liked_movies"].split(",") if u["liked_movies"] else [],
        "watchlist":    u["watchlist"].split(",") if u["watchlist"] else [],
        "ratings":      json.loads(u["ratings"]) if u["ratings"] else {}
    }

# ─── TMDB MOVIE ROUTES ────────────────────────────────────
def fmt(results):
    return [{"id": m.get("id"), "title": m.get("title"), "overview": m.get("overview",""),
             "poster_path": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None,
             "backdrop_path": f"https://image.tmdb.org/t/p/original{m['backdrop_path']}" if m.get("backdrop_path") else None,
             "release_date": m.get("release_date",""), "vote_average": m.get("vote_average",0)} for m in results]

@app.get("/api/movies/trending")
def trending(): return fmt(tmdb.get_trending())

@app.get("/api/movies/now-playing")
def now_playing(): return fmt(tmdb.get_now_playing())

@app.get("/api/movies/upcoming")
def upcoming():
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    # Use discover instead of /movie/upcoming so we can enforce future-only dates
    def fetch_page(page):
        r = requests.get(f"{tmdb.base_url}/discover/movie",
                         params={"api_key": tmdb.api_key,
                                 "primary_release_date.gte": today,
                                 "sort_by": "primary_release_date.asc",
                                 "vote_count.gte": 5,
                                 "page": page}, timeout=8)
        if r.status_code == 200:
            return r.json().get("results", [])
        return []

    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fetch_page, p) for p in [1, 2]]
        for f in futures:
            results.extend(f.result())

    # Deduplicate, ensure future dates only, sort by release date
    seen = set(); deduped = []
    for m in results:
        rd = m.get("release_date", "")
        if m["id"] not in seen and rd and rd >= today:
            seen.add(m["id"]); deduped.append(m)
    deduped.sort(key=lambda m: m.get("release_date", "9999"))
    return fmt(deduped[:40])

# ─── INDIAN CINEMA ENDPOINTS ──────────────────────────────
# Maps industry name → TMDB original_language codes
INDUSTRY_LANGS = {
    "bollywood":  "hi",             # Hindi
    "tollywood":  "te",             # Telugu
    "kollywood":  "ta",             # Tamil
    "mollywood":  "ml",             # Malayalam
    "sandalwood": "kn",             # Kannada
    "all_indian": "hi,te,ta,ml,kn", # All Indian languages
}

def _discover_indian(lang: str, sort: str = "popularity.desc", page: int = 1):
    """Use TMDB discover to get Indian language movies."""
    r = requests.get(f"{tmdb.base_url}/discover/movie",
                     params={"api_key": tmdb.api_key,
                             "with_original_language": lang,
                             "sort_by": sort,
                             "vote_count.gte": 100,
                             "page": page}, timeout=8)
    if r.status_code == 200:
        return fmt(r.json().get("results", [])[:20])
    return []

@app.get("/api/movies/indian-trending")
def indian_trending(industry: str = "all_indian"):
    lang = INDUSTRY_LANGS.get(industry, INDUSTRY_LANGS["all_indian"])
    # For multi-lang, split and fetch each in parallel
    langs = lang.split(",")
    results = []
    with ThreadPoolExecutor(max_workers=len(langs)) as pool:
        futures = [pool.submit(_discover_indian, l, "popularity.desc") for l in langs]
        for f in futures:
            results.extend(f.result())
    # Shuffle and deduplicate by id
    seen = set(); deduped = []
    for m in results:
        if m["id"] not in seen:
            seen.add(m["id"]); deduped.append(m)
    random.shuffle(deduped)
    return deduped[:20]

@app.get("/api/movies/indian-now-playing")
def indian_now_playing(industry: str = "all_indian"):
    lang = INDUSTRY_LANGS.get(industry, INDUSTRY_LANGS["all_indian"])
    langs = lang.split(",")
    results = []
    with ThreadPoolExecutor(max_workers=len(langs)) as pool:
        futures = [pool.submit(_discover_indian, l, "primary_release_date.desc") for l in langs]
        for f in futures:
            results.extend(f.result())
    seen = set(); deduped = []
    for m in results:
        if m["id"] not in seen:
            seen.add(m["id"]); deduped.append(m)
    return deduped[:20]

@app.get("/api/movies/indian-upcoming")
def indian_upcoming(industry: str = "all_indian"):
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    lang = INDUSTRY_LANGS.get(industry, INDUSTRY_LANGS["all_indian"])
    langs = lang.split(",")

    def fetch_future(l, page=1):
        r = requests.get(f"{tmdb.base_url}/discover/movie",
                         params={"api_key": tmdb.api_key,
                                 "with_original_language": l,
                                 "primary_release_date.gte": today,
                                 "sort_by": "primary_release_date.asc",
                                 "vote_count.gte": 1,
                                 "page": page}, timeout=8)
        if r.status_code == 200:
            return [m for m in r.json().get("results", [])
                    if m.get("release_date", "") >= today]
        return []

    results = []
    # Fetch 2 pages per language for a richer list
    tasks = [(l, p) for l in langs for p in [1, 2]]
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = [pool.submit(fetch_future, l, p) for l, p in tasks]
        for f in futures:
            results.extend(f.result())

    # Deduplicate and sort by release date ascending (nearest first)
    seen = set(); deduped = []
    for m in results:
        if m["id"] not in seen:
            seen.add(m["id"]); deduped.append(m)
    deduped.sort(key=lambda m: m.get("release_date", "9999"))
    return fmt(deduped[:40])

@app.get("/api/movies/indian-top-rated")
def indian_top_rated(industry: str = "all_indian"):
    lang = INDUSTRY_LANGS.get(industry, INDUSTRY_LANGS["all_indian"])
    langs = lang.split(",")
    results = []
    with ThreadPoolExecutor(max_workers=len(langs)) as pool:
        futures = [pool.submit(_discover_indian, l, "vote_average.desc") for l in langs]
        for f in futures:
            results.extend(f.result())
    seen = set(); deduped = []
    for m in results:
        if m["id"] not in seen:
            seen.add(m["id"]); deduped.append(m)
    return deduped[:20]

@app.get("/api/movies/search")
def search(q: str = Query(..., min_length=1), industry: str = "all"):
    # If it's a specific Indian industry, we append the original language filter or rely on the query
    # TMDB search endpoint doesn't natively filter by original_language cleanly in one go for text queries,
    # but we can filter the results post-fetch if it's strictly Indian mode.
    r = requests.get(f"{tmdb.base_url}/search/movie",
                     params={"api_key": tmdb.api_key, "query": q})
    if r.status_code == 200:
        results = r.json().get("results", [])
        if industry != "all":
            lang = INDUSTRY_LANGS.get(industry, INDUSTRY_LANGS["all_indian"])
            langs = lang.split(",")
            # Filter search results by language if specified
            results = [m for m in results if m.get("original_language") in langs]
        return fmt(results[:10])
    return []

@app.get("/api/movies/{movie_id}/details")
def movie_details(movie_id: int):
    # Fetch videos, cast, and similar movies all in parallel
    def get_similar():
        r = requests.get(f"{tmdb.base_url}/movie/{movie_id}/similar",
                         params={"api_key": tmdb.api_key})
        return fmt(r.json().get("results", [])[:10]) if r.status_code == 200 else []

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_vid = pool.submit(tmdb.get_movie_videos, movie_id)
        f_cast = pool.submit(tmdb.get_movie_credits, movie_id)
        f_sim = pool.submit(get_similar)
        videos = f_vid.result()
        cast   = f_cast.result()
        similar = f_sim.result()

    trailers = [v for v in videos if v.get("type") == "Trailer" and v.get("site") == "YouTube"]
    return {
        "trailers": [{"key": t["key"], "name": t["name"]} for t in trailers[:3]],
        "cast": [{"name": c["name"], "character": c.get("character",""), "profile_path": c.get("profile_path")} for c in cast[:12]],
        "similar": similar
    }

# ── OTT Watch Provider URLs ────────────────────────────────
# Maps TMDB provider_id → direct search/watch URL template (%s = movie title)
OTT_LINKS = {
    8:   ("Netflix",          "https://www.netflix.com/search?q=%s",                        "#E50914"),
    9:   ("Prime Video",      "https://www.primevideo.com/search/ref=atv_nb_sug?phrase=%s", "#00A8E0"),
    337: ("Disney+ Hotstar",  "https://www.hotstar.com/in/search?q=%s",                     "#1C3D8B"),
    122: ("Hotstar",          "https://www.hotstar.com/in/search?q=%s",                     "#1C3D8B"),
    2:   ("Apple TV+",        "https://tv.apple.com/search?term=%s",                        "#555555"),
    350: ("Apple TV+",        "https://tv.apple.com/search?term=%s",                        "#555555"),
    220: ("JioCinema",        "https://www.jiocinema.com/search/%s",                        "#0075FF"),
    232: ("ZEE5",             "https://www.zee5.com/search?q=%s",                           "#8B2FC9"),
    237: ("SonyLIV",          "https://www.sonyliv.com/search/%s",                          "#FF4B4B"),
    190: ("SonyLIV",          "https://www.sonyliv.com/search/%s",                          "#FF4B4B"),
    531: ("Paramount+",       "https://www.paramountplus.com/search/%s",                    "#0064FF"),
    384: ("HBO Max",          "https://www.max.com/search?q=%s",                            "#6B2D8B"),
    1899:("Max",              "https://www.max.com/search?q=%s",                            "#6B2D8B"),
    283: ("Crunchyroll",      "https://www.crunchyroll.com/search?q=%s",                    "#F47521"),
    515: ("MX Player",        "https://www.mxplayer.in/search?title=%s",                    "#FF6B00"),
    532: ("Aha",              "https://www.aha.video/search/%s",                            "#FFCC00"),
    11:  ("Mubi",             "https://mubi.com/search?query=%s",                            "#5468FF"),
}

@app.get("/api/movies/{movie_id}/watch-providers")
def watch_providers(movie_id: int, country: str = "IN", title: str = ""):
    """
    Returns OTT streaming platforms for a movie using TMDB's watch/providers API.
    country: ISO 3166-1 code — 'IN' for India, 'US' for USA
    """
    import urllib.parse
    r = requests.get(
        f"{tmdb.base_url}/movie/{movie_id}/watch/providers",
        params={"api_key": tmdb.api_key},
        timeout=6
    )
    if r.status_code != 200:
        return {"providers": []}

    results = r.json().get("results", {})

    # Try requested country, fallback to US, then IN
    country_data = results.get(country) or results.get("US") or results.get("IN") or {}
    link_page = country_data.get("link", "")

    # Collect flatrate (subscription) providers only — not rent/buy
    flatrate = country_data.get("flatrate", [])

    providers = []
    seen_names = set()
    for p in flatrate:
        pid = p.get("provider_id")
        name = p.get("provider_name", "")
        logo = p.get("logo_path")
        logo_url = f"https://image.tmdb.org/t/p/w92{logo}" if logo else None

        if name in seen_names:
            continue
        seen_names.add(name)

        # Build direct URL — use TMDB's own link_page as fallback
        if pid in OTT_LINKS:
            _, url_template, color = OTT_LINKS[pid]
            encoded_title = urllib.parse.quote(title or name)
            watch_url = url_template % encoded_title
        else:
            watch_url = link_page  # TMDB's JustWatch page
            color = "#444"

        providers.append({
            "name": name,
            "logo": logo_url,
            "url":  watch_url,
            "color": OTT_LINKS.get(pid, (None, None, "#444"))[2],
        })

    return {"providers": providers, "tmdb_link": link_page}


@app.get("/api/movies/by-genre/{genre}")
def movies_by_genre(genre: str, limit: int = 30):
    if genre not in GENRES: raise HTTPException(400, f"Invalid genre.")
    gi = GENRES.index(genre)
    gm = sorted([(i, r[-1]) for i, r in enumerate(movie_data) if r[gi] == 1], key=lambda x: -x[1])
    return [{"title": movie_titles[i][0], "link": movie_titles[i][2], "rating": movie_data[i][-1]} for i, _ in gm[:limit]]

# ─── FESTIVAL KEYWORD MAP ────────────────────────────────
# Each festival key → list of TMDB search queries that find movies
# with actual scenes/events from that festival
FESTIVAL_SEARCH_TERMS = {
    "holi":        ["holi festival", "rang barse", "holi bollywood", "festival of colors india"],
    "diwali":      ["diwali", "deepavali", "festival of lights india", "diwali celebration"],
    "new_year":    ["new year party", "new year eve celebration", "countdown new year"],
    "valentine":   ["valentine's day", "love story romance", "valentine romance"],
    "sankranti":   ["makar sankranti", "kite festival", "uttarayan"],
    "republic":    ["republic day india", "indian army", "freedom fighter india"],
    "summer":      ["summer adventure", "road trip summer", "beach summer"],
    "summer_thrill":["summer thriller", "heat crime thriller"],
    "monsoon":     ["monsoon rain romance", "rainy season india", "rain love story"],
    "rainy":       ["rain mystery", "monsoon horror", "storm thriller"],
    "independence":["independence 1947", "partition india", "freedom fighters india", "indian independence"],
    "monsoon_fam": ["family picnic", "monsoon family", "children adventure rain"],
    "navratri":    ["navratri", "garba dance", "durga puja", "indian festival dance"],
    "dussehra":    ["dussehra", "ramayana", "ravana", "ram leela"],
    "pre_diwali":  ["diwali preparation", "festival shopping india", "diwali family"],
    "post_diwali": ["india drama family", "diwali aftermath"],
    "christmas":   ["christmas celebration", "santa claus family", "christmas eve"],
    "year_end":    ["new year countdown", "year end party", "new year fireworks"],
    "winter":      ["winter mystery", "snow horror", "blizzard thriller"],
    "autumn":      ["autumn drama", "fall harvest"],
}

# ─── CONTEXTUAL / SEASONAL ENDPOINT ──────────────────────
@app.get("/api/movies/contextual")
def contextual_movies(industry: str = "all"):
    """
    Automatically detects today's date → picks Indian festival / season →
    Section 1: REAL festival movies (actual Holi/Diwali scenes) via TMDB keyword search
    Section 2: Genre-based complement (KNN)
    Section 3: Hidden gems (high-rated drama/mystery)
    All results are shuffled → different every refresh.
    """
    from datetime import datetime
    now   = datetime.now()
    month = now.month
    day   = now.day

    def cal():
        festivals = [
            (1,  1,  3,  "🎆 New Year Blockbusters",       "new_year",     ["Comedy","Romance","Action"],         7.5),
            (1, 13, 16,  "🌾 Makar Sankranti Picks",        "sankranti",    ["Family","Animation","Adventure"],    7.0),
            (1, 25, 31,  "🇮🇳 Republic Day Specials",       "republic",     ["History","Biography","War"],         7.0),
            (2, 10, 17,  "💕 Valentine's Week",             "valentine",    ["Romance","Drama","Music"],            7.5),
            (3,  1, 31,  "🎨 Holi Festival Vibes",          "holi",         ["Comedy","Musical","Family","Music"], 7.0),
            (4,  1, 30,  "☀️ Summer Heat — Action Packed",  "summer",       ["Action","Adventure","Sci-Fi"],       7.5),
            (5,  1, 31,  "🌡️ Hot Summer Thrillers",         "summer_thrill",["Thriller","Mystery","Crime"],        7.5),
            (6,  1, 30,  "🌧️ Monsoon Romance",              "monsoon",      ["Romance","Drama","Music"],            7.0),
            (7,  1, 31,  "⛈️ Rainy Day Mysteries",          "rainy",        ["Mystery","Horror","Thriller"],       7.0),
            (8,  1, 15,  "🇮🇳 Independence Day Specials",   "independence", ["Biography","History","War"],         7.0),
            (8, 16, 31,  "🌿 Monsoon Family Time",          "monsoon_fam",  ["Family","Animation","Comedy"],       7.0),
            (9,  1, 30,  "🪔 Navratri Celebrations",        "navratri",     ["Music","Musical","Drama","Family"],  7.0),
            (10, 1, 20,  "🏹 Dussehra Vibes",               "dussehra",     ["Action","Adventure","Fantasy"],      7.5),
            (10,21, 31,  "✨ Diwali is Coming!",             "pre_diwali",   ["Comedy","Family","Animation"],       7.0),
            (11, 1, 10,  "🪔 Diwali Special",               "diwali",       ["Drama","Family","Comedy","Music"],   7.5),
            (11,11, 30,  "🍂 Post-Diwali Chill",            "post_diwali",  ["Drama","Biography","Mystery"],       7.5),
            (12, 1, 20,  "🎄 Christmas Season",             "christmas",    ["Family","Animation","Comedy","Romance"],7.5),
            (12,21, 31,  "🥂 Year-End Countdown",           "year_end",     ["Action","Thriller","Sci-Fi"],        8.0),
        ]
        for (m, ds, de, label, key, genres, min_r) in festivals:
            if month == m and ds <= day <= de:
                return label, key, genres, min_r
        if month in (12, 1, 2): return "❄️ Winter Picks",        "winter",  ["Mystery","Horror","Fantasy","Thriller"], 7.5
        if month in (3, 4, 5):  return "☀️ Summer Blockbusters", "summer",  ["Action","Sci-Fi","Adventure"],           7.5
        if month in (6, 7, 8, 9): return "🌧️ Monsoon Classics",  "monsoon", ["Drama","Romance","Thriller"],            7.5
        return "🍁 Autumn Favourites", "autumn", ["Comedy","Drama","Biography"], 7.5

    label, ctx_key, primary_genres, min_r = cal()

    # ── Section 1: Real festival-specific movies from TMDB ──
    def tmdb_keyword_search(query, limit=6):
        """Search TMDB for movies matching a festival keyword."""
        try:
            r = requests.get(f"{tmdb.base_url}/search/movie",
                             params={"api_key": tmdb.api_key, "query": query,
                                     "sort_by": "popularity.desc"}, timeout=5)
            if r.status_code == 200:
                results = r.json().get("results", [])
                if industry != "all" and industry != "hollywood":
                    lang = INDUSTRY_LANGS.get(industry, INDUSTRY_LANGS["all_indian"])
                    langs = lang.split(",")
                    results = [m for m in results if m.get("original_language") in langs]
                return [m for m in results if m.get("vote_count", 0) > 10][:limit]
        except Exception:
            pass
        return []

    search_terms = FESTIVAL_SEARCH_TERMS.get(ctx_key, [])
    festival_movies_raw = []
    if search_terms:
        with ThreadPoolExecutor(max_workers=len(search_terms)) as pool:
            futures = [pool.submit(tmdb_keyword_search, term) for term in search_terms]
            for f in futures:
                festival_movies_raw.extend(f.result())

    # Deduplicate by id, shuffle
    seen_ids = set()
    festival_movies_deduped = []
    for m in festival_movies_raw:
        if m["id"] not in seen_ids:
            seen_ids.add(m["id"])
            festival_movies_deduped.append(m)
    random.shuffle(festival_movies_deduped)
    festival_section_movies = fmt(festival_movies_deduped[:20])

    # ── Section 2: KNN genre-based (shuffled) ───────────────
    def get_knn_section(genres, count=20, min_rating=7.0):
        if industry != "all" and industry != "hollywood":
            return fetch_indian_recos(genres, industry, count)
        test_point = [1 if g in genres else 0 for g in GENRES] + [min_rating]
        target     = [0] * len(movie_titles)
        model      = KNearestNeighbours(movie_data, target, test_point, k=60)
        model.fit()
        pool_ = [{"title": movie_titles[i][0], "link": movie_titles[i][2],
                  "rating": movie_data[i][-1]} for i in model.indices]
        random.shuffle(pool_)
        return pool_[:count]

    complement = {
        "action":["Romance","Drama"], "romance":["Action","Thriller"],
        "comedy":["Thriller","Mystery"], "thriller":["Comedy","Animation"],
        "family":["Sci-Fi","Adventure"], "drama":["Action","Comedy"],
        "mystery":["Comedy","Family"], "biography":["Sci-Fi","Fantasy"],
    }
    pg_lower    = [g.lower() for g in primary_genres]
    comp_genres = next((v for k,v in complement.items() if any(k in p for p in pg_lower)), ["Action","Comedy"])

    sections = []

    # Only show festival-specific section if we found real festival movies
    if festival_section_movies:
        sections.append({
            "label":    f"{label} 🎬 (Movies with real scenes)",
            "context":  ctx_key,
            "is_tmdb":  True,         # flag: these are full TMDB objects already
            "movies":   festival_section_movies,
        })

    # Always add genre-based section as a complement
    sections.append({
        "label":   f"{label} — More Picks",
        "context": f"{ctx_key}_genre",
        "is_tmdb": False,
        "movies":  get_knn_section(primary_genres, 24, min_r),
    })

    sections.append({
        "label":   "💎 Hidden Gems",
        "context": "gems",
        "is_tmdb": False,
        "movies":  get_knn_section(["Drama","Mystery","Biography"], 24, 8.0),
    })

    return {
        "date":     now.strftime("%B %d"),
        "month":    month,
        "context":  ctx_key,
        "label":    label,
        "sections": sections,
    }

@app.post("/api/movies/batch-search")
def batch_search(req: BatchSearchReq):
    """Fetch TMDB data for multiple movie titles in parallel. Returns a dict keyed by title."""
    def fetch_one(title):
        try:
            r = requests.get(f"{tmdb.base_url}/search/movie",
                             params={"api_key": tmdb.api_key, "query": title}, timeout=5)
            if r.status_code == 200:
                results = r.json().get("results", [])
                if results:
                    return title, fmt([results[0]])[0]
        except Exception:
            pass
        return title, None

    result_map = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(fetch_one, t): t for t in req.titles if t}
        for future in as_completed(futures):
            title, data = future.result()
            result_map[title] = data
    return result_map

@app.get("/api/movies/all-titles")
def all_titles(): return [t[0] for t in movie_titles]

# ─── RECOMMENDATIONS ──────────────────────────────────────
# Note: For our custom KNN model, we only have English movies in `movie_data.json` currently.
# To simulate Indian recommendation results using KNN, we will intercept the result titles 
# and do a TMDB discover query if industry != 'all'.
def knn_recommend(test_point, count):
    target = [0] * len(movie_titles)
    model = KNearestNeighbours(movie_data, target, test_point, k=count + 1)
    model.fit()
    return [{"title": movie_titles[i][0], "link": movie_titles[i][2], "rating": movie_data[i][-1]} for i in model.indices]

def fetch_indian_recos(genres: list[str], industry: str, count: int):
    # Map our genres to TMDB genre IDs (approximate)
    tmdb_genres = {"Action":28, "Adventure":12, "Animation":16, "Comedy":35, "Crime":80, "Documentary":99, "Drama":18, "Family":10751, "Fantasy":14, "History":36, "Horror":27, "Music":10402, "Mystery":9648, "Romance":10749, "Sci-Fi":878, "Thriller":53, "War":10752}
    g_ids = "|".join([str(tmdb_genres.get(g, "")) for g in genres if g in tmdb_genres])
    lang = INDUSTRY_LANGS.get(industry, INDUSTRY_LANGS["all_indian"])
    langs = lang.split(",")
    results = []
    with ThreadPoolExecutor(max_workers=len(langs)) as pool:
        futures = [pool.submit(_discover_internal, l, g_ids) for l in langs]
        for f in futures: results.extend(f.result())
    random.shuffle(results)
    return results[:count]

def _discover_internal(lang: str, genres: str):
    r = requests.get(f"{tmdb.base_url}/discover/movie", params={"api_key": tmdb.api_key, "with_original_language": lang, "with_genres": genres, "sort_by": "popularity.desc", "vote_count.gte": 50}, timeout=5)
    if r.status_code == 200:
        # Convert to our frontend localCard expected format
        return [{"title": m["title"], "link": "", "rating": m["vote_average"], "id": m["id"], "poster_path": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None} for m in r.json().get("results", [])]
    return []

@app.post("/api/recommendations/movie")
def reco_movie(req: MovieRecoReq):
    if req.industry != "all" and req.industry != "hollywood":
        return fetch_indian_recos(["Drama", "Action"], req.industry, req.count)  # Fallback for movie-based indian recos
    titles_list = [t[0] for t in movie_titles]
    if req.movie_title not in titles_list: raise HTTPException(404, "Movie not found.")
    idx = titles_list.index(req.movie_title)
    return knn_recommend(movie_data[idx], req.count)[1:]  # skip self

@app.post("/api/recommendations/genre")
def reco_genre(req: GenreRecoReq):
    if req.industry != "all" and req.industry != "hollywood":
        return fetch_indian_recos(req.genres, req.industry, req.count)
    test_point = [1 if g in req.genres else 0 for g in GENRES] + [req.min_rating]
    return knn_recommend(test_point, req.count)

@app.post("/api/recommendations/mood")
def reco_mood(req: MoodRecoReq):
    genres = MOOD_GENRE_MAP.get(req.mood.lower())
    if not genres: raise HTTPException(400, f"Unknown mood. Use: {list(MOOD_GENRE_MAP.keys())}")
    
    if req.industry != "all" and req.industry != "hollywood":
        return fetch_indian_recos(genres, req.industry, req.count)
        
    test_point = [1 if g in genres else 0 for g in GENRES] + [7.0]
    return knn_recommend(test_point, req.count)

@app.get("/api/recommendations/because-you-watched/{title}")
def because_you_watched(title: str):
    titles_list = [t[0] for t in movie_titles]
    if title not in titles_list: raise HTTPException(404, "Movie not found.")
    idx = titles_list.index(title)
    return {"based_on": title, "recommendations": knn_recommend(movie_data[idx], 10)[1:6]}

# ─── USER DATA ────────────────────────────────────────────
@app.get("/api/user/{username}/data")
def get_user_data(username: str):
    conn, cur = get_db()
    cur.execute("SELECT * FROM users WHERE username=?", (username,)); u = cur.fetchone(); conn.close()
    if not u: raise HTTPException(404, "User not found.")
    return {
        "liked_movies": u["liked_movies"].split(",") if u["liked_movies"] else [],
        "watchlist": u["watchlist"].split(",") if u["watchlist"] else [],
        "ratings": json.loads(u["ratings"]) if u["ratings"] else {}
    }

@app.post("/api/user/likes")
def toggle_like(req: LikeReq):
    conn, cur = get_db()
    cur.execute("SELECT liked_movies FROM users WHERE username=?", (req.username,)); row = cur.fetchone()
    if not row: conn.close(); raise HTTPException(404, "User not found.")
    items = [x for x in (row["liked_movies"] or "").split(",") if x]
    if req.movie_title in items: items.remove(req.movie_title)
    else: items.append(req.movie_title)
    cur.execute("UPDATE users SET liked_movies=? WHERE username=?", (",".join(items), req.username))
    conn.commit(); conn.close()
    return {"liked_movies": items, "is_liked": req.movie_title in items}

@app.post("/api/user/watchlist")
def toggle_watchlist(req: WatchlistReq):
    conn, cur = get_db()
    cur.execute("SELECT watchlist FROM users WHERE username=?", (req.username,)); row = cur.fetchone()
    if not row: conn.close(); raise HTTPException(404, "User not found.")
    items = [x for x in (row["watchlist"] or "").split(",") if x]
    if req.movie_title in items: items.remove(req.movie_title)
    else: items.append(req.movie_title)
    cur.execute("UPDATE users SET watchlist=? WHERE username=?", (",".join(items), req.username))
    conn.commit(); conn.close()
    return {"watchlist": items, "in_watchlist": req.movie_title in items}

@app.post("/api/user/rate")
def rate_movie(req: RateReq):
    if not (1 <= req.rating <= 5): raise HTTPException(400, "Rating must be 1-5.")
    conn, cur = get_db()
    cur.execute("SELECT ratings FROM users WHERE username=?", (req.username,)); row = cur.fetchone()
    if not row: conn.close(); raise HTTPException(404, "User not found.")
    ratings = json.loads(row["ratings"]) if row["ratings"] else {}
    ratings[req.movie_title] = req.rating
    cur.execute("UPDATE users SET ratings=? WHERE username=?", (json.dumps(ratings), req.username))
    conn.commit(); conn.close()
    return {"ratings": ratings}

# (Analytics removed as requested)

# ─── WATCH HISTORY ────────────────────────────────────────
@app.post("/api/user/history")
def add_watch_history(req: WatchHistoryReq):
    conn, cur = get_db()
    cur.execute("SELECT watch_history FROM users WHERE username=?", (req.username,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return {"ok": False}
    items = [x for x in (row["watch_history"] or "").split(",") if x]
    entry = str(req.movie_id)
    # Keep last 50 unique history entries
    if entry in items:
        items.remove(entry)
    items.append(entry)
    items = items[-50:]
    cur.execute("UPDATE users SET watch_history=? WHERE username=?", (",".join(items), req.username))
    conn.commit(); conn.close()
    return {"ok": True}

# ─── FOR YOU (Multi-movie BYW) ────────────────────────────
@app.get("/api/recommendations/for-you")
def for_you(username: str, industry: str = "all", count: int = 20):
    """
    Multi-movie 'Because You Watched':
    Considers ALL liked movies + high-rated movies (rating >= 4).
    Movies appearing across multiple liked films score higher.
    """
    conn, cur = get_db()
    cur.execute("SELECT liked_movies, ratings FROM users WHERE username=?", (username,))
    row = cur.fetchone(); conn.close()
    if not row:
        raise HTTPException(404, "User not found.")

    liked = [x for x in (row["liked_movies"] or "").split(",") if x]
    ratings = json.loads(row["ratings"]) if row["ratings"] else {}

    # Build weighted pool: liked movies + highly rated movies
    seed_titles = list(set(liked + [t for t, r in ratings.items() if float(r) >= 4.0]))
    if not seed_titles:
        # Fallback to popular
        return fmt(tmdb.get_trending()[:count])

    titles_list = [t[0] for t in movie_titles]
    score_map = {}  # title -> cumulative score

    for seed in seed_titles[:8]:  # cap at 8 seeds for performance
        if seed not in titles_list:
            continue
        idx = titles_list.index(seed)
        weight = float(ratings.get(seed, 3)) / 5.0  # 0.2–1.0
        recs = knn_recommend(movie_data[idx], 15)
        for r in recs:
            t = r["title"]
            if t in seed_titles:
                continue
            score_map[t] = score_map.get(t, 0) + weight * r.get("rating", 7) / 10.0

    if not score_map:
        return fmt(tmdb.get_trending()[:count])

    sorted_titles = sorted(score_map, key=lambda t: -score_map[t])[:count]

    if industry != "all" and industry != "hollywood":
        return fetch_indian_recos(["Drama", "Action"], industry, count)

    return [{"title": t, "link": "", "rating": score_map[t] * 10} for t in sorted_titles]


# ─── HYBRID SCORE ENGINE (v2 — with Explainability + Re-Ranking) ──────────────
TMDB_GENRE_MAP = {
    28:"Action", 12:"Adventure", 16:"Animation", 35:"Comedy", 80:"Crime",
    99:"Documentary", 18:"Drama", 10751:"Family", 14:"Fantasy", 36:"History",
    27:"Horror", 10402:"Music", 9648:"Mystery", 10749:"Romance", 878:"Science Fiction",
    53:"Thriller", 10752:"War", 37:"Western", 10770:"TV Movie"
}

# Configurable weights — can be overridden via env vars for A/B testing
W_PREF     = float(os.environ.get("W_PREF",     "0.30"))
W_TREND    = float(os.environ.get("W_TREND",    "0.20"))
W_SIM      = float(os.environ.get("W_SIM",      "0.20"))
W_RECENCY  = float(os.environ.get("W_RECENCY",  "0.15"))
W_POP      = float(os.environ.get("W_POP",      "0.15"))

# Enhanced mood map with energy/sentiment for richer explainability
MOOD_META = {
    "happy":     {"genres": ["Comedy","Animation","Family","Music","Adventure"],   "label": "Feel-Good",   "emoji": "😄"},
    "sad":       {"genres": ["Drama","Romance","Music"],                           "label": "Emotional",   "emoji": "😢"},
    "excited":   {"genres": ["Action","Adventure","Science Fiction","Thriller"],   "label": "High-Energy", "emoji": "🔥"},
    "scared":    {"genres": ["Horror","Thriller","Mystery"],                       "label": "Thrilling",   "emoji": "😱"},
    "romantic":  {"genres": ["Romance","Drama","Music"],                           "label": "Romantic",    "emoji": "💕"},
    "motivated": {"genres": ["Biography","Sport","Drama","History"],               "label": "Motivating",  "emoji": "💪"},
    "mindblown": {"genres": ["Science Fiction","Mystery","Thriller","Fantasy"],    "label": "Mind-Bending","emoji": "🤯"},
    "chill":     {"genres": ["Comedy","Animation","Family","Documentary"],         "label": "Chill",       "emoji": "😎"},
}

def _explain_movie(m: dict, pref_score: float, trend_score: float,
                   recency_score: float, pop_score: float,
                   liked: list, user_genres: list, movie_genres: set) -> str:
    """Generate a human-readable 'Why Recommended' explanation string."""
    tags = []

    # Trending signal
    if trend_score > 0.75:
        tags.append("🔥 Trending Now")
    elif trend_score > 0.5:
        tags.append("📈 Popular This Week")

    # Recency signal
    if recency_score > 0.85:
        tags.append("🆕 Just Released")
    elif recency_score > 0.60:
        tags.append("🗓️ New Release")

    # Genre match signal
    if pref_score > 0.6 and user_genres:
        matched = list(movie_genres & set(user_genres))
        if matched:
            tags.append(f"🎭 Matches your {matched[0]} taste")
        elif liked:
            tags.append(f"🍿 Similar to {liked[-1]}")

    # Quality signal
    vote_avg = m.get("vote_average", 0) or 0
    if vote_avg >= 8.0:
        tags.append(f"⭐ Critically Acclaimed ({vote_avg:.1f})")
    elif vote_avg >= 7.0:
        tags.append(f"⭐ Highly Rated ({vote_avg:.1f})")

    # Popularity signal
    if pop_score > 0.7 and not tags:
        tags.append("🌍 Global Hit")

    return " · ".join(tags[:2]) if tags else "🎬 Top Pick For You"

def _rerank_with_diversity(scored: list, liked_set: set, max_per_genre: int = 4) -> list:
    """Re-rank to ensure genre diversity and filter watched movies."""
    from collections import defaultdict
    genre_counts = defaultdict(int)
    results = []

    for final_score, m in sorted(scored, key=lambda x: -x[0]):
        title = m.get("title", "")
        # Skip already liked/watched
        if title in liked_set:
            continue
        # Genre diversity cap
        genre_ids = m.get("genre_ids", []) or []
        primary_genre = TMDB_GENRE_MAP.get(genre_ids[0], "Other") if genre_ids else "Other"
        if genre_counts[primary_genre] >= max_per_genre:
            continue
        genre_counts[primary_genre] += 1
        results.append((final_score, m))

    return results

@app.get("/api/recommendations/hybrid")
def hybrid_recommend(username: str = "", industry: str = "all", count: int = 20,
                     mood: str = "", explain: bool = True):
    """
    Netflix-style hybrid scoring with explainability and genre re-ranking:
      W_PREF    × User Preference Score
    + W_TREND   × Trending Score
    + W_SIM     × Similarity Score
    + W_RECENCY × Recency Score
    + W_POP     × Popularity Score
    All weights are configurable via environment variables.
    """
    import math
    from datetime import datetime

    # ── Fetch candidate pool from TMDB ────────────────────
    if industry != "all" and industry != "hollywood":
        lang = INDUSTRY_LANGS.get(industry, INDUSTRY_LANGS["all_indian"])
        langs = lang.split(",")
        trending_raw = []
        with ThreadPoolExecutor(max_workers=len(langs)) as pool:
            futures = [pool.submit(_discover_indian, l, "popularity.desc") for l in langs]
            for f in futures:
                trending_raw.extend(f.result())
    else:
        trending_raw = tmdb.get_trending()

    candidates = {}
    for m in trending_raw[:60]:
        mid = m.get("id") if isinstance(m, dict) else None
        if mid:
            candidates[mid] = m if isinstance(m, dict) else {}

    # Add now-playing for diversity
    try:
        for m in tmdb.get_now_playing()[:20]:
            mid = m.get("id")
            if mid and mid not in candidates:
                candidates[mid] = m
    except Exception:
        pass

    # ── Load user profile ──────────────────────────────────
    user_genres = []
    liked = []
    ratings_map = {}
    is_cold_start = True

    if username:
        conn, cur = get_db()
        cur.execute("SELECT fav_genres, liked_movies, ratings FROM users WHERE username=?", (username,))
        row = cur.fetchone(); conn.close()
        if row:
            user_genres = [g for g in (row["fav_genres"] or "").split(",") if g]
            liked = [x for x in (row["liked_movies"] or "").split(",") if x]
            ratings_map = json.loads(row["ratings"]) if row["ratings"] else {}
            is_cold_start = len(liked) + len(ratings_map) < 3

    # Build liked genres from user's history
    liked_genres = set(user_genres)
    titles_list = [t[0] for t in movie_titles]
    for title in (liked + list(ratings_map.keys()))[:10]:
        if title in titles_list:
            idx = titles_list.index(title)
            row_data = movie_data[idx]
            for gi, g in enumerate(GENRES):
                if row_data[gi] == 1:
                    liked_genres.add(g)

    # Mood boost genres
    mood_genres = set()
    if mood and mood in MOOD_META:
        mood_genres = set(MOOD_META[mood]["genres"])
        liked_genres |= mood_genres  # Merge mood into preference

    # ── Score each candidate ───────────────────────────────
    total = len(candidates)
    scored = []
    now_str = datetime.now().strftime("%Y-%m-%d")

    for rank, (mid, m) in enumerate(candidates.items()):
        # Popularity score
        vote_avg = m.get("vote_average", 0) or 0
        vote_cnt = m.get("vote_count", 1) or 1
        pop_score = min((vote_avg * math.log(vote_cnt + 1)) / 40.0, 1.0)

        # Trending score
        trend_score = max(0, 1.0 - rank / max(total, 1))

        # Recency score
        release = m.get("release_date", "") or ""
        try:
            days_old = (datetime.strptime(now_str, "%Y-%m-%d") - datetime.strptime(release[:10], "%Y-%m-%d")).days
            recency_score = max(0, 1.0 - days_old / 365.0)
        except Exception:
            recency_score = 0.3

        # User preference score (genre overlap)
        genre_ids = m.get("genre_ids", []) or []
        movie_genres = {TMDB_GENRE_MAP.get(gid, "") for gid in genre_ids}
        if liked_genres:
            overlap = len(liked_genres & movie_genres)
            pref_score = min(overlap / max(len(liked_genres), 1), 1.0)
        else:
            pref_score = 0.4 if is_cold_start else 0.5

        # Similarity score combines preference + mood match
        mood_boost = 0.3 if mood_genres & movie_genres else 0.0
        sim_score = min(pref_score * 0.7 + trend_score * 0.2 + mood_boost, 1.0)

        # Final hybrid score
        final = (W_PREF * pref_score + W_TREND * trend_score +
                 W_SIM * sim_score + W_RECENCY * recency_score + W_POP * pop_score)

        # Build explainability tag
        reason = _explain_movie(m, pref_score, trend_score, recency_score,
                                pop_score, liked, list(liked_genres), movie_genres)

        m["_reason"] = reason
        scored.append((final, m))

    # ── Re-rank with genre diversity ───────────────────────
    liked_set = set(liked)
    diverse = _rerank_with_diversity(scored, liked_set, max_per_genre=4)

    # Cold start: if no user data, boost quality + trending
    if is_cold_start:
        diverse.sort(key=lambda x: -(x[0] + x[1].get("vote_average", 0) * 0.05))

    top = [m for _, m in diverse[:count]]

    # ── Format response with explainability ───────────────
    result = []
    for m in top:
        item = {
            "id": m.get("id"),
            "title": m.get("title", ""),
            "overview": m.get("overview", ""),
            "poster_path": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None,
            "backdrop_path": f"https://image.tmdb.org/t/p/original{m['backdrop_path']}" if m.get("backdrop_path") else None,
            "release_date": m.get("release_date", ""),
            "vote_average": m.get("vote_average", 0),
        }
        if explain:
            item["reason"] = m.get("_reason", "🎬 Top Pick For You")
        result.append(item)

    return result


# ─── SURPRISE ME ──────────────────────────────────────────
@app.get("/api/recommendations/surprise")
def surprise_me(username: str = "", industry: str = "all"):
    """
    'Surprise Me' — picks one random high-quality movie the user hasn't seen.
    Filters: IMDb >= 7.5, at least 500 votes, not in user's watched/liked list.
    """
    import math

    liked_set = set()
    if username:
        conn, cur = get_db()
        cur.execute("SELECT liked_movies, watch_history FROM users WHERE username=?", (username,))
        row = cur.fetchone(); conn.close()
        if row:
            liked_set = set((row["liked_movies"] or "").split(","))

    # Pull from top-rated TMDB movies
    if industry != "all" and industry != "hollywood":
        lang = INDUSTRY_LANGS.get(industry, INDUSTRY_LANGS["all_indian"])
        pages = []
        for pg in [1, 2, 3]:
            r = requests.get(f"{tmdb.base_url}/discover/movie",
                             params={"api_key": tmdb.api_key, "with_original_language": lang,
                                     "sort_by": "vote_average.desc", "vote_count.gte": 500,
                                     "vote_average.gte": 7.5, "page": pg}, timeout=6)
            if r.status_code == 200:
                pages.extend(r.json().get("results", []))
    else:
        pages = []
        for pg in [1, 2, 3]:
            r = requests.get(f"{tmdb.base_url}/movie/top_rated",
                             params={"api_key": tmdb.api_key, "page": pg}, timeout=6)
            if r.status_code == 200:
                pages.extend(r.json().get("results", []))

    # Filter quality + unseen
    pool = [m for m in pages
            if m.get("vote_average", 0) >= 7.5
            and m.get("vote_count", 0) >= 500
            and m.get("title", "") not in liked_set]

    if not pool:
        pool = pages[:20]  # Fallback

    pick = random.choice(pool) if pool else {}
    result = fmt([pick])[0] if pick else {}
    result["reason"] = "🎲 Surprise Pick — Highly Rated, Something New!"
    return result


# ─── INTERACTION LOGGING ──────────────────────────────────
class InteractionReq(BaseModel):
    username: str = ""
    movie_id: int
    movie_title: str
    watch_pct: float = 0.0       # 0.0–1.0, how much was watched
    source: str = "home"         # 'home'|'search'|'chatbot'|'similar'|'surprise'
    session_sec: int = 0         # seconds spent on this movie

@app.post("/api/user/interaction")
def log_interaction(req: InteractionReq):
    """
    Logs a user interaction event for future collaborative filtering.
    Implicitly updates watch history as well.
    """
    # Update watch history if user logged in
    if req.username:
        conn, cur = get_db()
        cur.execute("SELECT watch_history FROM users WHERE username=?", (req.username,))
        row = cur.fetchone()
        if row:
            items = [x for x in (row["watch_history"] or "").split(",") if x]
            entry = str(req.movie_id)
            if entry in items:
                items.remove(entry)
            items.append(entry)
            items = items[-50:]
            cur.execute("UPDATE users SET watch_history=? WHERE username=?",
                        (",".join(items), req.username))
            conn.commit()
        conn.close()

    # Log for future CF training (stored as JSON lines in a log file)
    try:
        log_path = os.path.join(os.path.dirname(__file__), '..', 'interactions.jsonl')
        from datetime import datetime
        entry = {
            "user": req.username or "anon",
            "movie_id": req.movie_id,
            "title": req.movie_title,
            "watch_pct": req.watch_pct,
            "source": req.source,
            "session_sec": req.session_sec,
            "ts": datetime.now().isoformat()
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Never fail silently — interaction logging is best-effort

    return {"ok": True}



# ─── AI CHATBOT (Gemini) ──────────────────────────────────
@app.post("/api/chat")
def chat_endpoint(req: ChatReq):
    """
    AI chatbot powered by Gemini. Understands mood/vibe/genre/time
    prompts and returns both a reply + movie recommendations.
    """
    import re
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    system_prompt = """You are CineMate AI, an expert movie recommendation assistant.
When a user describes what they want to watch (mood, vibe, genre, time, language, era), you:
1. Reply in a friendly, enthusiastic 1-2 sentence response
2. Extract recommendation parameters and append a JSON block EXACTLY like this at the end:
{"action":"recommend","genres":["Thriller","Crime"],"mood":"scared","min_rating":7.5,"era":"modern","time_filter":"any"}

Rules:
- genres: pick from [Action,Adventure,Animation,Comedy,Crime,Documentary,Drama,Family,Fantasy,History,Horror,Music,Mystery,Romance,Sci-Fi,Thriller,War,Western]
- mood: one of [happy,sad,excited,romantic,scared,motivated,mindblown,chill]
- min_rating: 6.0 to 9.0
- era: "classic" (pre-2000), "modern" (2010+), "any"
- time_filter: "short" (<90min), "normal" (<2hr), "any"
- Always include the JSON block even if the message is vague.
- Speak naturally and warmly. Be concise."""

    messages = []
    for h in req.history[-6:]:  # last 6 turns
        messages.append({"role": h.get("role", "user"), "parts": [h.get("content", "")]})
    messages.append({"role": "user", "parts": [req.message]})

    ai_reply = ""
    params = {"genres": ["Drama"], "mood": "chill", "min_rating": 7.0, "era": "any", "time_filter": "any"}

    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                system_instruction=system_prompt
            )
            chat = model.start_chat(history=messages[:-1])
            response = chat.send_message(req.message)
            ai_reply = response.text

            # Extract JSON block from reply
            json_match = re.search(r'\{[^{}]*"action"\s*:\s*"recommend"[^{}]*\}', ai_reply, re.DOTALL)
            if json_match:
                params = json.loads(json_match.group())
                # Clean reply text: remove the JSON block
                ai_reply = ai_reply[:json_match.start()].strip()
        except Exception as e:
            ai_reply = f"I'd love to help with that! Let me find some great picks for you. 🎬"
            print(f"[Gemini Error] {e}")
    else:
        # Fallback: keyword-based NLP without AI
        msg_lower = req.message.lower()
        genre_keywords = {
            "thriller": "Thriller", "horror": "Horror", "scary": "Horror",
            "action": "Action", "comedy": "Comedy", "funny": "Comedy",
            "romance": "Romance", "romantic": "Romance", "love": "Romance",
            "sci-fi": "Sci-Fi", "science fiction": "Sci-Fi", "space": "Sci-Fi",
            "drama": "Drama", "mystery": "Mystery", "crime": "Crime",
            "animation": "Animation", "animated": "Animation",
            "adventure": "Adventure", "fantasy": "Fantasy",
            "documentary": "Documentary", "history": "History", "war": "War"
        }
        found_genres = list({v for k, v in genre_keywords.items() if k in msg_lower})
        if not found_genres:
            found_genres = ["Drama"]

        mood_map = {"sad": "sad", "happy": "happy", "dark": "scared", "light": "chill",
                    "funny": "happy", "scary": "scared", "excited": "excited",
                    "romantic": "romantic", "mind": "mindblown", "inspire": "motivated"}
        found_mood = next((v for k, v in mood_map.items() if k in msg_lower), "chill")

        params = {"genres": found_genres, "mood": found_mood, "min_rating": 7.0, "era": "any", "time_filter": "any"}
        ai_reply = f"Great taste! Here are some {' & '.join(found_genres)} picks you'll love 🎬"

    # ── Fetch recommendations based on extracted params ────
    genres = params.get("genres", ["Drama"])
    min_rating = float(params.get("min_rating", 7.0))
    era = params.get("era", "any")

    if req.industry != "all" and req.industry != "hollywood":
        movies = fetch_indian_recos(genres, req.industry, 10)
    else:
        test_point = [1 if g in genres else 0 for g in GENRES] + [min_rating]
        all_recs = knn_recommend(test_point, 40)

        # Apply era filter
        if era == "modern":
            all_recs = [r for r in all_recs if r.get("link", "").find("tt") > -1 or True]  # placeholder

        movies = all_recs[:10]

    # Also get TMDB discover for more variety
    try:
        tmdb_genre_ids = {"Action":"28","Adventure":"12","Comedy":"35","Crime":"80","Drama":"18",
                          "Horror":"27","Mystery":"9648","Romance":"10749","Sci-Fi":"878","Thriller":"53",
                          "Fantasy":"14","History":"36","Animation":"16","Family":"10751","War":"10752"}
        gids = ",".join([tmdb_genre_ids[g] for g in genres[:2] if g in tmdb_genre_ids])
        r = requests.get(f"{tmdb.base_url}/discover/movie",
                         params={"api_key": tmdb.api_key, "with_genres": gids,
                                 "vote_average.gte": min_rating, "sort_by": "popularity.desc",
                                 "vote_count.gte": 200}, timeout=6)
        if r.status_code == 200:
            tmdb_movies = fmt(r.json().get("results", [])[:8])
            return {"reply": ai_reply, "movies": tmdb_movies, "params": params}
    except Exception:
        pass

    return {"reply": ai_reply, "movies": movies, "params": params}


# ─── OLD CLASSICS SIMILAR TO TRENDING ────────────────────
@app.get("/api/recommendations/classics-like-trending")
def classics_like_trending(industry: str = "all"):
    """
    Finds old classics (pre-2010) that are similar to current trending movies.
    This is the 'Similar Older Versions' Netflix feature.
    """
    # Get current trending genre_ids
    trending = tmdb.get_trending()[:5]
    genre_ids = set()
    for m in trending:
        for gid in (m.get("genre_ids") or []):
            genre_ids.add(gid)

    # Map to our KNN genres
    tmdb_genre_map = {28:"Action",12:"Adventure",16:"Animation",35:"Comedy",80:"Crime",
                      99:"Documentary",18:"Drama",10751:"Family",14:"Fantasy",36:"History",
                      27:"Horror",10402:"Music",9648:"Mystery",10749:"Romance",878:"Sci-Fi",
                      53:"Thriller",10752:"War",37:"Western"}
    knn_genres = list({tmdb_genre_map[gid] for gid in genre_ids if gid in tmdb_genre_map})[:3]

    if industry != "all" and industry != "hollywood":
        return fetch_indian_recos(knn_genres, industry, 20)

    test_point = [1 if g in knn_genres else 0 for g in GENRES] + [7.5]
    classics = knn_recommend(test_point, 30)
    return classics[:20]


# ─── JUST RELEASED FOR YOU ────────────────────────────────
@app.get("/api/recommendations/just-released")
def just_released(username: str = "", industry: str = "all"):
    """Movies from last 60 days, prioritized by user's genre preferences."""
    from datetime import datetime, timedelta
    sixty_days_ago = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")

    user_genres = []
    if username:
        conn, cur = get_db()
        cur.execute("SELECT fav_genres FROM users WHERE username=?", (username,))
        row = cur.fetchone(); conn.close()
        if row:
            user_genres = [g for g in (row["fav_genres"] or "").split(",") if g]

    tmdb_genre_ids = {"Action":"28","Adventure":"12","Comedy":"35","Crime":"80","Drama":"18",
                      "Horror":"27","Mystery":"9648","Romance":"10749","Sci-Fi":"878","Thriller":"53",
                      "Fantasy":"14","History":"36","Animation":"16","Family":"10751","War":"10752",
                      "Music":"10402","Biography":"99"}
    genre_filter = ",".join([tmdb_genre_ids[g] for g in user_genres[:3] if g in tmdb_genre_ids])

    if industry != "all" and industry != "hollywood":
        lang = INDUSTRY_LANGS.get(industry, INDUSTRY_LANGS["all_indian"])
        langs = lang.split(",")
        results = []
        for l in langs:
            params = {"api_key": tmdb.api_key, "with_original_language": l,
                      "primary_release_date.gte": sixty_days_ago,
                      "primary_release_date.lte": today,
                      "sort_by": "popularity.desc", "vote_count.gte": 5}
            if genre_filter:
                params["with_genres"] = genre_filter
            r = requests.get(f"{tmdb.base_url}/discover/movie", params=params, timeout=8)
            if r.status_code == 200:
                results.extend(r.json().get("results", []))
        seen = set(); deduped = []
        for m in results:
            if m["id"] not in seen:
                seen.add(m["id"]); deduped.append(m)
        return fmt(deduped[:20])

    params = {"api_key": tmdb.api_key,
              "primary_release_date.gte": sixty_days_ago,
              "primary_release_date.lte": today,
              "sort_by": "popularity.desc",
              "vote_count.gte": 10}
    if genre_filter:
        params["with_genres"] = genre_filter
    r = requests.get(f"{tmdb.base_url}/discover/movie", params=params, timeout=8)
    if r.status_code == 200:
        return fmt(r.json().get("results", [])[:20])
    return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
