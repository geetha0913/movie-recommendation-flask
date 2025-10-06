from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the movies dataset
movies = pd.read_csv('movies.csv')

# Function to combine features for recommendation
def combine_features(row):
    return ' '.join([
        str(row['genres']),
        str(row['keywords']),
        str(row['tagline']),
        str(row['cast']),
        str(row['director'])
    ])

# Create a new column for combined features
movies['combined_features'] = movies.apply(combine_features, axis=1)

# Create TF-IDF matrix from combined features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Recommendation route
@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie']

    if movie_name not in movies['title'].values:
        return render_template('result.html', movie=movie_name, recommendations=None)

    # Get movie index
    movie_idx = movies[movies['title'] == movie_name].index[0]

    # Get similarity scores for this movie
    sim_scores = list(enumerate(cosine_sim[movie_idx]))

    # Sort movies based on similarity score (exclude the movie itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    # Get the titles of the top 5 similar movies
    recommended_movies = [movies['title'].iloc[i[0]] for i in sim_scores]

    return render_template('result.html', movie=movie_name, recommendations=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)

