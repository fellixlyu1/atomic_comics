from flask import Flask, render_template, request, redirect, url_for, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv

app = Flask(__name__)
app.secret_key = 'your_secret_key'

comics_dict = {}

with open("csv_files/Marvel_Comics.csv") as comics_csv:
    comics_data = list(csv.DictReader(comics_csv))
    for row in comics_data:
        comics_dict[row["issue_title"]] = row


@app.route("/", methods=['GET'])
def index():
    return render_template('atomic_comics.html', comics=comics_data)


@app.route('/comic_details')
def comic_details():
    title = request.args.get('title')
    selected_comic = next((comic for comic in comics_data if comic['issue_title'] == title), None)
    if selected_comic:
        recommendations = get_recommendation(title)
        return render_template('comic_details.html', comic=selected_comic, recommendations=recommendations)


@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    title = request.form.get('title')
    if title in comics_dict:
        session_cart = session.get('cart', [])
        session_cart.append(title)
        session['cart'] = session_cart
    return redirect(url_for('cart'))


@app.route('/cart')
def cart():
    session_cart = session.get('cart', [])
    cart_items = [comics_dict[title] for title in session_cart]
    for item in cart_items:
        item['total_price'] = float(item['Price'].replace(' $', ''))
    total_cart_price = sum(item['total_price'] for item in cart_items)
    return render_template('cart.html', cart_items=cart_items, total_cart_price=total_cart_price)


@app.route('/clear_all', methods=['POST'])
def clear_all():
    session['cart'] = []
    return redirect(url_for('cart'))


@app.route('/purchase', methods=['POST'])
def purchase():
    return render_template('purchase.html')


def get_recommendation(issue_title, num_recommendations=6):
    issue_descriptions = [entry["issue_description"] for entry in comics_dict.values()]
    vectorize = TfidfVectorizer()
    matrix = vectorize.fit_transform(issue_descriptions)
    index = list(comics_dict.keys()).index(issue_title)
    cosine_sim = cosine_similarity(matrix[index], matrix).flatten()

    similar_comics_indices = cosine_sim.argsort()[:-num_recommendations - 1:-1]
    top_recommendations = [list(comics_dict.keys())[i] for i in similar_comics_indices if i != index]

    return top_recommendations


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
