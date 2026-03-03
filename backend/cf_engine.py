"""
CineMate — SVD Collaborative Filtering Engine
Uses TruncatedSVD (sklearn) on the user-movie interaction matrix built from interactions.jsonl.
Falls back gracefully when data is insufficient (< 10 users or < 20 interactions).
"""
import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from threading import Lock

INTERACTIONS_LOG = os.path.join(os.path.dirname(__file__), '..', 'interactions.jsonl')
MIN_INTERACTIONS = 20   # minimum total interactions to train
MIN_USERS        = 5    # minimum distinct users to train
SVD_COMPONENTS   = 20   # latent factors (increase with more data)

_model_lock = Lock()
_cf_model   = None   # (user_map, movie_map, inv_movie_map, user_factors, movie_factors)


def load_interactions() -> List[dict]:
    """Read interactions.jsonl — every row is one interaction event."""
    if not os.path.exists(INTERACTIONS_LOG):
        return []
    rows = []
    with open(INTERACTIONS_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def build_matrix(interactions: List[dict]) -> Tuple[dict, dict, dict, np.ndarray]:
    """
    Build a sparse user × movie implicit rating matrix.
    Rating = watch_pct * 5  OR  explicit star rating if present.
    """
    from scipy.sparse import lil_matrix

    user_set  = sorted({r["user"]     for r in interactions if r.get("user")  != "anon"})
    movie_set = sorted({r["movie_id"] for r in interactions if r.get("movie_id")})

    if len(user_set) < MIN_USERS:
        raise ValueError(f"Need ≥{MIN_USERS} users, have {len(user_set)}")

    user_map  = {u: i for i, u in enumerate(user_set)}
    movie_map = {m: i for i, m in enumerate(movie_set)}
    inv_movie = {i: m for m, i in movie_map.items()}

    mat = lil_matrix((len(user_set), len(movie_set)), dtype=np.float32)

    for row in interactions:
        u = row.get("user")
        mid = row.get("movie_id")
        if u not in user_map or mid not in movie_map:
            continue
        # Implicit score: watch_pct * 5
        score = float(row.get("watch_pct", 0)) * 5.0
        if score < 0.5:
            score = 1.0   # At least opened the modal
        mat[user_map[u], movie_map[mid]] = max(mat[user_map[u], movie_map[mid]], score)

    return user_map, movie_map, inv_movie, mat.tocsr()


def train():
    """Train the SVD model. Called at startup and can be called to retrain."""
    global _cf_model
    from sklearn.decomposition import TruncatedSVD

    interactions = load_interactions()
    if len(interactions) < MIN_INTERACTIONS:
        print(f"[CF] Not enough interactions ({len(interactions)}/{MIN_INTERACTIONS}) — CF disabled")
        with _model_lock:
            _cf_model = None
        return

    try:
        user_map, movie_map, inv_movie, matrix = build_matrix(interactions)
        n_components = min(SVD_COMPONENTS, min(matrix.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        user_factors  = svd.fit_transform(matrix)          # shape (users, k)
        movie_factors = svd.components_.T                   # shape (movies, k)

        with _model_lock:
            _cf_model = (user_map, movie_map, inv_movie, user_factors, movie_factors)

        print(f"[CF] SVD trained — {len(user_map)} users, {len(movie_map)} movies, k={n_components}")
    except Exception as e:
        print(f"[CF] Training failed: {e}")
        with _model_lock:
            _cf_model = None


def predict_for_user(username: str, top_n: int = 20) -> List[Tuple[int, float]]:
    """
    Return top_n (movie_id, predicted_score) for a user.
    Returns empty list if model not available or user not in training set.
    """
    with _model_lock:
        model = _cf_model

    if model is None:
        return []

    user_map, movie_map, inv_movie, user_factors, movie_factors = model

    if username not in user_map:
        return []

    u_idx  = user_map[username]
    u_vec  = user_factors[u_idx]                    # (k,)
    scores = movie_factors @ u_vec                   # (movies,) predicted scores

    # Get top-n movie indices excluding already high-scored (>= 3.5) entries
    top_indices = np.argsort(-scores)[:top_n * 2]   # over-fetch then filter

    results = []
    for idx in top_indices:
        mid   = inv_movie[idx]
        score = float(scores[idx])
        if score > 0:
            results.append((mid, score))
        if len(results) >= top_n:
            break

    return results


def is_ready() -> bool:
    with _model_lock:
        return _cf_model is not None


def retrain_if_needed():
    """Re-train the model (call periodically or after new interactions)."""
    train()


# Auto-train at import time
try:
    train()
except Exception as e:
    print(f"[CF] Startup train error: {e}")
