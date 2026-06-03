"""
app.py  —  CineMatch Flask Backend
====================================
Run:   python app.py
Open:  http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
from recommender import MovieRecommender, MOVIES, RATINGS_DATA
import json

app = Flask(__name__)
rec = MovieRecommender(MOVIES, RATINGS_DATA)

EMOJIS = {1:"🦇",2:"🌀",3:"📖",4:"🚗",5:"🚀",6:"🐶",7:"🛸",8:"🎵",
          9:"🔥",10:"🤖",11:"🏠",12:"💌",13:"🔪",14:"👁️",15:"🚢",16:"👰"}
YEARS  = {1:2008,2:2010,3:2004,4:2015,5:2014,6:2014,7:2016,8:2016,
          9:1995,10:2017,11:2019,12:2005,13:2019,14:2017,15:1997,16:2014}


@app.route("/")
def index():
    movies_dict = {
        str(mid): {"title": info["title"], "genres": info["genres"],
                   "emoji": EMOJIS.get(mid,"🎬"), "year": YEARS.get(mid,"")}
        for mid, info in MOVIES.items()
    }
    return render_template("index.html", movies_json=json.dumps(movies_dict))


@app.route("/api/recommend", methods=["POST"])
def recommend():
    try:
        body         = request.get_json()
        user_ratings = {int(k): int(v) for k, v in body.get("ratings", {}).items()}
        cf_weight    = float(body.get("cf_weight", 0.6))
        cb_weight    = float(body.get("cb_weight", 0.4))
        top_n        = int(body.get("top_n", 6))
        if not user_ratings:
            return jsonify({"error": "No ratings provided"}), 400
        df = rec.recommend(user_ratings, top_n=top_n, cf_weight=cf_weight, cb_weight=cb_weight)
        return jsonify({"recommendations": df.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/similar/<int:movie_id>")
def similar(movie_id):
    try:
        df = rec.similar_movies(movie_id, top_n=6)
        return jsonify({"similar": df.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  🎬 CineMatch — AI Movie Recommender")
    print("  Open → http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
