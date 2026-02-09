from flask import Flask, render_template, request
from model import predict_genre, dataset_stats

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""

    if request.method == "POST":
        plot = request.form["plot"]
        prediction = predict_genre(plot)

    total, genre_counts = dataset_stats()

    return render_template(
        "index.html",
        prediction=prediction,
        total=total,
        genre_counts=genre_counts
    )

if __name__ == "__main__":
    app.run(debug=True)