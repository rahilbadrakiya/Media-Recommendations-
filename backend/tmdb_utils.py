import requests
import os

class TMDBClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p/w500"

    def get_movie_details_by_title(self, title):
        """Search for a movie by title and return its details."""
        search_url = f"{self.base_url}/search/movie"
        params = {
            "api_key": self.api_key,
            "query": title
        }
        response = requests.get(search_url, params=params)
        if response.status_code == 200:
            results = response.json().get("results")
            if results:
                return results[0] # Return the most relevant result
        return None

    def get_poster_url(self, poster_path):
        """Construct a full poster URL from a TMDB poster path."""
        if poster_path:
            return f"{self.image_base_url}{poster_path}"
        return None

    def get_now_playing(self, page=1):
        """Fetch 'Now Playing' movies."""
        url = f"{self.base_url}/movie/now_playing"
        params = {
            "api_key": self.api_key,
            "page": page
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json().get("results", [])
        return []

    def get_trending(self, time_window="day"):
        """Fetch trending movies."""
        url = f"{self.base_url}/trending/movie/{time_window}"
        params = {"api_key": self.api_key}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json().get("results", [])
        return []

    def get_upcoming(self, page=1):
        """Fetch 'Upcoming' movies."""
        url = f"{self.base_url}/movie/upcoming"
        params = {
            "api_key": self.api_key,
            "page": page
        }
        # Use a region lock to avoid spoilers or irrelevant results if needed,
        # but for now, general upcoming is fine.
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json().get("results", [])
        return []

    def get_movie_videos(self, movie_id):
        """Fetch trailers and other videos for a movie."""
        url = f"{self.base_url}/movie/{movie_id}/videos"
        params = {"api_key": self.api_key}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json().get("results", [])
        return []

    def get_movie_credits(self, movie_id):
        """Fetch cast and crew for a movie."""
        url = f"{self.base_url}/movie/{movie_id}/credits"
        params = {"api_key": self.api_key}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json().get("cast", [])
        return []
