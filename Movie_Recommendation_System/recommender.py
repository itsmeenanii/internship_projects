"""
Movie Recommendation System
============================
Techniques used:
  1. Collaborative Filtering  — user-based (cosine similarity on rating matrix)
  2. Content-Based Filtering  — genre overlap (Jaccard similarity)
  3. Hybrid Recommender       — weighted blend of both scores
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ── Sample Dataset ──────────────────────────────────────────────────────────
# In a real project you would load the MovieLens CSV here:
#   df = pd.read_csv("ratings.csv")
# We use a compact built-in dataset so the project runs without downloads.

MOVIES = {
    1:  {"title": "The Dark Knight",     "genres": ["Action", "Thriller"]},
    2:  {"title": "Inception",           "genres": ["Sci-Fi", "Thriller"]},
    3:  {"title": "The Notebook",        "genres": ["Romance", "Drama"]},
    4:  {"title": "Mad Max: Fury Road",  "genres": ["Action", "Sci-Fi"]},
    5:  {"title": "Interstellar",        "genres": ["Sci-Fi", "Drama"]},
    6:  {"title": "John Wick",           "genres": ["Action", "Thriller"]},
    7:  {"title": "Arrival",             "genres": ["Sci-Fi", "Drama"]},
    8:  {"title": "La La Land",          "genres": ["Romance", "Drama", "Musical"]},
    9:  {"title": "Heat",                "genres": ["Action", "Thriller", "Crime"]},
    10: {"title": "Blade Runner 2049",   "genres": ["Sci-Fi", "Drama"]},
    11: {"title": "Parasite",            "genres": ["Drama", "Thriller"]},
    12: {"title": "Pride & Prejudice",   "genres": ["Romance", "Drama"]},
    13: {"title": "Knives Out",          "genres": ["Thriller", "Comedy"]},
    14: {"title": "Get Out",             "genres": ["Horror", "Thriller"]},
    15: {"title": "Titanic",             "genres": ["Romance", "Drama"]},
    16: {"title": "Gone Girl",           "genres": ["Thriller", "Drama"]},
}

# Simulated historical ratings: rows = users, cols = movie IDs (0 = not rated)
RATINGS_DATA = {
    "Alice":   {1:5, 2:4, 4:5, 6:4, 9:3},
    "Bob":     {2:5, 5:5, 7:4, 10:5, 11:3},
    "Carol":   {3:5, 8:4, 12:5, 15:5},
    "David":   {1:4, 6:5, 9:5, 13:3, 14:4},
    "Eve":     {2:3, 5:4, 7:5, 10:4, 16:5},
    "Frank":   {3:4, 8:5, 12:4, 15:3, 11:4},
    "Grace":   {1:5, 4:4, 6:5, 9:4, 16:3},
    "Henry":   {2:4, 7:5, 10:3, 11:5, 13:4},
    "Iris":    {3:3, 8:4, 15:5, 12:3, 5:2},
    "Jack":    {14:5, 13:4, 16:5, 11:4, 9:3},
}


class MovieRecommender:
    """
    Hybrid movie recommender combining:
    - User-based Collaborative Filtering (cosine similarity)
    - Content-Based Filtering (Jaccard genre similarity)
    """

    def __init__(self, movies: dict, ratings_data: dict):
        self.movies = movies
        self.ratings_data = ratings_data
        self.all_genres = sorted({g for m in movies.values() for g in m["genres"]})
        self._build_matrices()

    # ── Matrix Building ──────────────────────────────────────────────────────

    def _build_matrices(self):
        """Build the user-movie rating matrix and genre feature matrix."""
        movie_ids = sorted(self.movies.keys())
        users = list(self.ratings_data.keys())

        # User-Movie Rating Matrix (users × movies)
        rating_matrix = pd.DataFrame(0.0, index=users, columns=movie_ids)
        for user, ratings in self.ratings_data.items():
            for mid, rating in ratings.items():
                rating_matrix.loc[user, mid] = float(rating)

        self.rating_matrix = rating_matrix

        # Genre Feature Matrix (movies × genres) — one-hot encoded
        genre_matrix = pd.DataFrame(0, index=movie_ids, columns=self.all_genres)
        for mid, info in self.movies.items():
            for g in info["genres"]:
                genre_matrix.loc[mid, g] = 1

        self.genre_matrix = genre_matrix

        # Precompute cosine similarity between users
        self.user_similarity = pd.DataFrame(
            cosine_similarity(rating_matrix.values),
            index=users,
            columns=users,
        )

        # Precompute cosine similarity between movies (content-based)
        self.movie_content_sim = pd.DataFrame(
            cosine_similarity(genre_matrix.values),
            index=movie_ids,
            columns=movie_ids,
        )

    # ── Collaborative Filtering ──────────────────────────────────────────────

    def collaborative_score(self, user_ratings: dict, top_k_users: int = 5) -> pd.Series:
        """
        User-based collaborative filtering.
        Steps:
          1. Build a rating vector for the new user.
          2. Compute cosine similarity vs. all historical users.
          3. Find the top-K most similar users.
          4. Predict ratings as a weighted average of similar users' ratings.
        """
        movie_ids = sorted(self.movies.keys())
        new_vec = np.array([float(user_ratings.get(mid, 0)) for mid in movie_ids])

        # Cosine similarity: new user vs all historical users
        hist_matrix = self.rating_matrix.values            # shape: (n_users, n_movies)
        norms = np.linalg.norm(hist_matrix, axis=1)        # each user's L2 norm
        new_norm = np.linalg.norm(new_vec)

        similarities = np.dot(hist_matrix, new_vec) / (norms * new_norm + 1e-9)

        # Top-K similar users
        top_k_idx = np.argsort(similarities)[::-1][:top_k_users]
        top_k_sims = similarities[top_k_idx]
        top_k_ratings = hist_matrix[top_k_idx]             # shape: (K, n_movies)

        # Weighted average prediction
        sim_sum = np.sum(np.abs(top_k_sims)) + 1e-9
        predicted = np.dot(top_k_sims, top_k_ratings) / sim_sum

        scores = pd.Series(predicted, index=movie_ids)

        # Zero out already-rated movies
        for mid in user_ratings:
            scores[mid] = 0.0

        return scores

    # ── Content-Based Filtering ──────────────────────────────────────────────

    def content_based_score(self, user_ratings: dict) -> pd.Series:
        """
        Content-based filtering using genre Jaccard similarity.
        Scores each unrated movie by how similar its genres are
        to the genres of movies the user liked (rated >= 4).
        """
        liked = [mid for mid, r in user_ratings.items() if r >= 4]
        movie_ids = sorted(self.movies.keys())

        if not liked:
            return pd.Series(0.0, index=movie_ids)

        # Average genre vector of liked movies
        liked_vec = self.genre_matrix.loc[liked].mean(axis=0).values  # shape: (n_genres,)

        scores = {}
        for mid in movie_ids:
            if mid in user_ratings:
                scores[mid] = 0.0
                continue
            movie_vec = self.genre_matrix.loc[mid].values
            # Jaccard: intersection / union on binary vectors
            intersection = np.minimum(liked_vec, movie_vec).sum()
            union = np.maximum(liked_vec, movie_vec).sum()
            scores[mid] = intersection / (union + 1e-9)

        return pd.Series(scores)

    # ── Hybrid Recommender ───────────────────────────────────────────────────

    def recommend(
        self,
        user_ratings: dict,
        top_n: int = 5,
        cf_weight: float = 0.6,
        cb_weight: float = 0.4,
    ) -> pd.DataFrame:
        """
        Hybrid recommendation combining collaborative + content-based scores.

        Parameters
        ----------
        user_ratings : dict  {movie_id: rating (1-5)}
        top_n        : int   number of recommendations to return
        cf_weight    : float weight for collaborative filtering score
        cb_weight    : float weight for content-based score

        Returns
        -------
        DataFrame with columns: title, genres, cf_score, cb_score, hybrid_score
        """
        cf_scores = self.collaborative_score(user_ratings)
        cb_scores = self.content_based_score(user_ratings)

        # Normalise both score series to [0, 1]
        def normalise(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn + 1e-9)

        cf_norm = normalise(cf_scores)
        cb_norm = normalise(cb_scores)

        hybrid = cf_weight * cf_norm + cb_weight * cb_norm

        result = []
        for mid in sorted(self.movies.keys()):
            if mid not in user_ratings:
                result.append({
                    "movie_id":     mid,
                    "title":        self.movies[mid]["title"],
                    "genres":       ", ".join(self.movies[mid]["genres"]),
                    "cf_score":     round(cf_scores[mid], 4),
                    "cb_score":     round(cb_scores[mid], 4),
                    "hybrid_score": round(hybrid[mid], 4),
                })

        df = pd.DataFrame(result).sort_values("hybrid_score", ascending=False)
        return df.head(top_n).reset_index(drop=True)

    # ── Similar Movies (Content-Based) ───────────────────────────────────────

    def similar_movies(self, movie_id: int, top_n: int = 5) -> pd.DataFrame:
        """Return movies most similar to a given movie by genre cosine similarity."""
        sims = self.movie_content_sim[movie_id].drop(movie_id).sort_values(ascending=False)
        result = []
        for mid, sim in sims.head(top_n).items():
            result.append({
                "movie_id":   mid,
                "title":      self.movies[mid]["title"],
                "genres":     ", ".join(self.movies[mid]["genres"]),
                "similarity": round(sim, 4),
            })
        return pd.DataFrame(result)

    # ── User Similarity Report ────────────────────────────────────────────────

    def user_similarity_report(self) -> pd.DataFrame:
        """Return the full user-user cosine similarity matrix."""
        return self.user_similarity.round(4)
