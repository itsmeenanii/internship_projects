# 🎬 CineMatch — Movie Recommendation System

An AI-powered movie recommendation system built with Python.
Implements **Collaborative Filtering**, **Content-Based Filtering**, and a **Hybrid approach**.

---

## 📁 Project Structure

```
movie_recommender/
├── recommender.py      # Core AI algorithms
├── main.py             # Demo runner + interactive mode
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🧠 AI Techniques Used

### 1. User-Based Collaborative Filtering
- Builds a **User × Movie rating matrix**
- Computes **cosine similarity** between users
- Predicts ratings as a **weighted average** of top-K similar users' ratings

### 2. Content-Based Filtering
- Encodes movie genres as **one-hot feature vectors**
- Computes **cosine similarity** between movies
- Recommends movies similar in genre to what the user liked (rated ≥ 4★)

### 3. Hybrid Recommender
- Combines both scores with configurable weights (default: 60% CF + 40% CB)
- Normalises both score series to [0, 1] before blending

---

## ⚙️ Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the demo
python main.py
```

---

## 💡 How to Use

In `main.py`, pass your own ratings dict to get recommendations:

```python
from recommender import MovieRecommender, MOVIES, RATINGS_DATA

rec = MovieRecommender(MOVIES, RATINGS_DATA)

my_ratings = {
    1: 5,   # The Dark Knight  ★★★★★
    2: 4,   # Inception        ★★★★
    6: 5,   # John Wick        ★★★★★
}

recs = rec.recommend(my_ratings, top_n=5)
print(recs)
```

---

## 📊 Key Functions

| Function | Description |
|---|---|
| `recommend(user_ratings, top_n)` | Hybrid recommendations |
| `collaborative_score(user_ratings)` | CF scores only |
| `content_based_score(user_ratings)` | CB scores only |
| `similar_movies(movie_id, top_n)` | Movies similar to a given movie |
| `user_similarity_report()` | Full user-user cosine similarity matrix |

---

## 🚀 Extend This Project

- Load real data: [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/)
- Add matrix factorization (SVD) using `surprise` library
- Build a Flask web API on top of `recommender.py`
- Add the HTML frontend (`movie_recommendation_system.html`) as the UI

---

*Built for AI Internship Project — Hybrid Movie Recommendation System*
