"""
main.py — Run and demo the Movie Recommendation System
========================================================
Run:  python main.py
"""

from recommender import MovieRecommender, MOVIES, RATINGS_DATA


def print_header(title: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_recommendations(df, label="Recommendations"):
    print(f"\n  {'#':<4} {'Title':<25} {'Genres':<30} {'Score'}")
    print("  " + "-" * 70)
    for i, row in df.iterrows():
        score = row.get("hybrid_score", row.get("similarity", ""))
        genres = row["genres"][:28] + ".." if len(row["genres"]) > 30 else row["genres"]
        print(f"  {i+1:<4} {row['title']:<25} {genres:<30} {score:.4f}")


def demo():
    recommender = MovieRecommender(MOVIES, RATINGS_DATA)

    # ── Demo 1: Hybrid Recommendations ──────────────────────────────────────
    print_header("DEMO 1 — Hybrid Recommendations (Action/Thriller fan)")

    user_ratings = {
        1: 5,   # The Dark Knight    ★★★★★
        6: 4,   # John Wick          ★★★★
        9: 5,   # Heat               ★★★★★
        2: 3,   # Inception          ★★★
    }

    print("\n  Your ratings:")
    for mid, r in user_ratings.items():
        print(f"    {'★' * r}{'☆' * (5-r)}  {MOVIES[mid]['title']}")

    recs = recommender.recommend(user_ratings, top_n=5, cf_weight=0.6, cb_weight=0.4)
    print("\n  Top 5 recommendations:")
    print_recommendations(recs)
    print(f"\n  CF weight: 60%  |  CB weight: 40%  |  Algorithm: Hybrid")

    # ── Demo 2: Content-Based only (Sci-Fi fan) ──────────────────────────────
    print_header("DEMO 2 — Content-Based Filtering (Sci-Fi fan)")

    scifi_ratings = {
        2: 5,   # Inception          ★★★★★
        5: 5,   # Interstellar       ★★★★★
        7: 4,   # Arrival            ★★★★
    }

    print("\n  Your ratings:")
    for mid, r in scifi_ratings.items():
        print(f"    {'★' * r}{'☆' * (5-r)}  {MOVIES[mid]['title']}")

    recs_cb = recommender.recommend(scifi_ratings, top_n=5, cf_weight=0.0, cb_weight=1.0)
    print("\n  Top 5 recommendations (content-based only):")
    print_recommendations(recs_cb)

    # ── Demo 3: Collaborative Filtering only ─────────────────────────────────
    print_header("DEMO 3 — Collaborative Filtering only (Romance fan)")

    romance_ratings = {
        3:  5,  # The Notebook       ★★★★★
        8:  4,  # La La Land         ★★★★
        15: 5,  # Titanic            ★★★★★
    }

    print("\n  Your ratings:")
    for mid, r in romance_ratings.items():
        print(f"    {'★' * r}{'☆' * (5-r)}  {MOVIES[mid]['title']}")

    recs_cf = recommender.recommend(romance_ratings, top_n=5, cf_weight=1.0, cb_weight=0.0)
    print("\n  Top 5 recommendations (collaborative filtering only):")
    print_recommendations(recs_cf)

    # ── Demo 4: Similar Movies ───────────────────────────────────────────────
    print_header("DEMO 4 — Content-Based: Movies Similar to 'Inception'")

    sim = recommender.similar_movies(movie_id=2, top_n=5)
    print_recommendations(sim)

    # ── Demo 5: User Similarity Matrix ──────────────────────────────────────
    print_header("DEMO 5 — User-User Cosine Similarity Matrix")
    sim_matrix = recommender.user_similarity_report()
    print()
    print(sim_matrix.to_string())

    # ── Demo 6: Interactive Mode ─────────────────────────────────────────────
    print_header("DEMO 6 — Interactive: Rate Movies & Get Recommendations")
    print("\n  Available movies:")
    for mid, info in MOVIES.items():
        print(f"    [{mid:>2}] {info['title']:<25}  {', '.join(info['genres'])}")

    print("\n  Enter your ratings (movie_id rating), e.g.  '1 5'")
    print("  Type 'done' when finished.\n")

    interactive_ratings = {}
    while True:
        try:
            inp = input("  > ").strip()
            if inp.lower() in ("done", "q", "exit", ""):
                break
            parts = inp.split()
            if len(parts) != 2:
                print("  Format: <movie_id> <rating 1-5>")
                continue
            mid, rating = int(parts[0]), int(parts[1])
            if mid not in MOVIES:
                print(f"  Movie ID {mid} not found.")
                continue
            if not (1 <= rating <= 5):
                print("  Rating must be between 1 and 5.")
                continue
            interactive_ratings[mid] = rating
            print(f"  Saved: {'★' * rating}{'☆' * (5-rating)}  {MOVIES[mid]['title']}")
        except (ValueError, KeyboardInterrupt):
            break

    if interactive_ratings:
        print("\n  Generating your personalised recommendations...")
        recs_interactive = recommender.recommend(interactive_ratings, top_n=5)
        print("\n  Your top 5 recommendations:")
        print_recommendations(recs_interactive)
    else:
        print("\n  No ratings entered. Skipping interactive recommendations.")

    print("\n" + "=" * 60)
    print("  Done! Thank you for using CineMatch.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    demo()
