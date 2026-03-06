"""
CineMate — SentenceTransformer Embedding Pre-Computation Script
Run ONCE to pre-compute embeddings for all movies in the dataset.

Usage:
    python backend/embed_movies.py

Output:
    backend/Data/movie_embeddings.npy   — float32 array, shape (N, 384)
    backend/Data/movie_embed_index.json — maps array row → movie title
"""
import os
import sys
import json
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
TITLES_FILE  = os.path.join(DATA_DIR, "movie_titles.json")
DATA_FILE    = os.path.join(DATA_DIR, "movie_data.json")
EMBED_FILE   = os.path.join(DATA_DIR, "movie_embeddings.npy")
INDEX_FILE   = os.path.join(DATA_DIR, "movie_embed_index.json")

GENRES = ['Action','Adventure','Animation','Biography','Comedy','Crime','Documentary','Drama',
          'Family','Fantasy','Film-Noir','Game-Show','History','Horror','Music','Musical',
          'Mystery','News','Reality-TV','Romance','Sci-Fi','Short','Sport','Thriller','War','Western']


def build_text(title: str, genre_vec: list) -> str:
    """Create a rich text representation of a movie for embedding."""
    genres = [GENRES[i] for i, v in enumerate(genre_vec[:len(GENRES)]) if v == 1]
    rating = genre_vec[len(GENRES)] if len(genre_vec) > len(GENRES) else 0
    return (f"{title}. Genre: {', '.join(genres) or 'Unknown'}. "
            f"Rating: {rating:.1f}/10. "
            f"This is a {' and '.join(genres[:2])} movie.")


def compute_embeddings():
    print("Loading movie data...")
    with open(TITLES_FILE, encoding="utf-8") as f:
        titles = json.load(f)   # list of [title, ...]
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)     # list of feature vectors

    print(f"Found {len(titles)} movies.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        print("Run: pip install sentence-transformers")
        sys.exit(1)

    print("Loading model: all-MiniLM-L6-v2 (87MB, fast)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build texts
    texts = []
    index = []
    for i, (title_entry, vec) in enumerate(zip(titles, data)):
        title = title_entry[0] if isinstance(title_entry, list) else str(title_entry)
        text  = build_text(title, vec)
        texts.append(text)
        index.append(title)

    print(f"Computing embeddings for {len(texts)} movies...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True   # L2-normalized for cosine similarity via dot product
    )

    np.save(EMBED_FILE, embeddings.astype(np.float32))
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f)

    print(f"\n✅ Done!")
    print(f"   Embeddings: {EMBED_FILE}  ({embeddings.shape})")
    print(f"   Index:      {INDEX_FILE}")
    print(f"\nNow restart the backend — semantic similarity is active.")


if __name__ == "__main__":
    compute_embeddings()
