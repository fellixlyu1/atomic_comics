from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv

comics_dict = {}

with open("/home/fellixlyu1/PycharmProjects/atomic_comics/csv_files/Marvel_Comics.csv") as comics_csv:
    comics_data = list(csv.DictReader(comics_csv))
    for row in comics_data:
        comics_dict[row["issue_title"]] = row


def test_1(issue_title):
    issue_descriptions = [entry["issue_description"] for entry in comics_dict.values()]
    vectorize = TfidfVectorizer()
    matrix = vectorize.fit_transform(issue_descriptions)
    index = list(comics_dict.keys()).index(issue_title)
    cosine_sim = cosine_similarity(matrix[index], matrix).flatten()
    return cosine_sim


def test_2(issue_title, num_recommendations=6):
    run_test_1 = test_1(issue_title)
    index = list(comics_dict.keys()).index(issue_title)
    similar_comics_indices = run_test_1.argsort()[:-num_recommendations - 1:-1]
    top_recommendations = [list(comics_dict.keys())[i] for i in similar_comics_indices if i != index]
    return top_recommendations


title = input("Enter the issue: ")
print(test_2(title))
