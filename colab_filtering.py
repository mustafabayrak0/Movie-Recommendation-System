
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

class CollaborativeFilteringModel:
    def __init__(self, ratings_file, movies_file):
        self.ratings_file = ratings_file
        self.movies_file = movies_file
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.model = None

    def load_data(self):
        # Load the dataset
        ratings = self.ratings_file
        movies = self.movies_file

        # Merge ratings and movies data
        data = pd.merge(ratings, movies, on='movieId')

        # Encode user and movie IDs
        data['user'] = self.user_encoder.fit_transform(data['userId'].values)
        data['movie'] = self.movie_encoder.fit_transform(data['movieId'].values)

        return data

    def recommend_movies(self, user_id, model_file='/home/mustafa/Desktop/Courses/Data-Mining/Project/15/collaborative_filtering_model.h5', top_n=10):
        # Load the saved model
        model = self.get_model()

        # Load the full dataset
        data = self.load_data()

        # Check if the user_id is in the dataset
        if user_id not in data['userId'].unique():
            print("User ID not found in the dataset.")
            return

        # Filter data for the given user ID
        user_data = data[data['userId'] == user_id]

        # Encode the given user ID
        encoded_user_id = self.user_encoder.transform([user_id])

        # Get all unique movie IDs from the dataset and encode them
        all_movies = data['movieId'].unique()
        encoded_movies = self.movie_encoder.transform(all_movies)

        # Predict ratings for all movies for this user
        predicted_ratings = model.predict([np.array([encoded_user_id[0]] * len(all_movies)), encoded_movies])

        # Create a DataFrame with movie IDs and their predicted ratings
        movie_ratings = pd.DataFrame({
            'movieId': all_movies,
            'predicted_rating': predicted_ratings.flatten()
        })

        # Exclude movies that the user has already rated
        rated_movies = user_data['movieId'].unique()
        recommendations = movie_ratings[~movie_ratings['movieId'].isin(rated_movies)]

        # Sort the movies based on predicted ratings and select the top N
        top_recommendations = recommendations.sort_values(by='predicted_rating', ascending=False).head(top_n)

        # Translate back to movie titles
        movies_df = self.movies_file
        top_recommendations = top_recommendations.merge(movies_df, on='movieId')

        return top_recommendations[['movieId', 'title', 'predicted_rating']]

    def get_model(self, model_file='/home/mustafa/Desktop/Courses/Data-Mining/Project/15/collaborative_filtering_model.h5'):
        # Get the model
        return load_model('/home/mustafa/Desktop/Courses/Data-Mining/Project/15/collaborative_filtering_model.h5')
