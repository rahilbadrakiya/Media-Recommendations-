"""
CineMate — TTL In-Memory Cache
Redis-compatible interface: swap with redis-py by changing the `get`/`set`/`delete` calls.
Environment: set REDIS_URL=redis://localhost:6379 to switch to real Redis.
"""
import time
import os
import json
from threading import Lock

# ── Try Redis first, fallback to in-process TTL cache ────────────────────────
_redis_client = None
REDIS_URL = os.environ.get("REDIS_URL", "")

if REDIS_URL:
    try:
        import redis
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()
        print(f"[Cache] Connected to Redis at {REDIS_URL}")
    except Exception as e:
        print(f"[Cache] Redis unavailable ({e}) — using in-process TTL cache")
        _redis_client = None
else:
    print("[Cache] No REDIS_URL — using in-process TTL cache (set REDIS_URL for production)")


class _TTLCache:
    """Thread-safe in-process TTL cache. Swappable with Redis."""

    def __init__(self):
        self._store: dict = {}
        self._lock = Lock()

    def get(self, key: str):
        with self._lock:
            entry = self._store.get(key)
            if entry and time.time() < entry["exp"]:
                return entry["val"]
            if entry:
                del self._store[key]   # Expired
        return None

    def set(self, key: str, value, ttl: int = 300):
        with self._lock:
            self._store[key] = {"val": value, "exp": time.time() + ttl}

    def delete(self, key: str):
        with self._lock:
            self._store.pop(key, None)

    def flush_prefix(self, prefix: str):
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                del self._store[k]

    def stats(self) -> dict:
        with self._lock:
            now = time.time()
            alive = [k for k, v in self._store.items() if v["exp"] > now]
        return {"total_keys": len(self._store), "alive_keys": len(alive)}


# ── Public interface (Redis or in-process) ────────────────────────────────────
class Cache:
    def get(self, key: str):
        if _redis_client:
            val = _redis_client.get(key)
            return json.loads(val) if val else None
        return _ttl.get(key)

    def set(self, key: str, value, ttl: int = 300):
        if _redis_client:
            _redis_client.setex(key, ttl, json.dumps(value))
        else:
            _ttl.set(key, value, ttl)

    def delete(self, key: str):
        if _redis_client:
            _redis_client.delete(key)
        else:
            _ttl.delete(key)

    def get_or_set(self, key: str, fn, ttl: int = 300):
        """Decorator-style: return cached value or compute + cache it."""
        val = self.get(key)
        if val is not None:
            return val
        val = fn()
        self.set(key, val, ttl)
        return val

    def stats(self) -> dict:
        if _redis_client:
            info = _redis_client.info("memory")
            return {"backend": "redis", "used_memory_human": info.get("used_memory_human")}
        return {"backend": "in-process", **_ttl.stats()}


_ttl = _TTLCache()
cache = Cache()

# ── TTL configuration (seconds) ───────────────────────────────────────────────
TTL = {
    "trending":        3600,     # 1 hour
    "now_playing":     1800,     # 30 min
    "hybrid":          900,      # 15 min
    "content_similar": 86400,    # 24 hours
    "watch_providers": 43200,    # 12 hours
    "search":          300,      # 5 min
    "top_rated":       7200,     # 2 hours
    "contextual":      1800,     # 30 min
    "for_you":         600,      # 10 min
    "cf_scores":       1800,     # 30 min
}
