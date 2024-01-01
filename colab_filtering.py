
import numpy as np
from keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam


path = "/home/mustafa/Desktop/Courses/Data-Mining/Project/15"

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
    
    def create_model(self, num_users, num_movies, embedding_size=50):
        user_input = Input(shape=(1,), name='user_input')
        movie_input = Input(shape=(1,), name='movie_input')

        user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, input_length=1)(user_input)
        movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_size, input_length=1)(movie_input)

        user_flatten = Flatten()(user_embedding)
        movie_flatten = Flatten()(movie_embedding)

        merged = Concatenate()([user_flatten, movie_flatten])
        dense1 = Dense(4, activation='tanh')(merged)
        output = Dense(1, activation='sigmoid')(dense1)

        model = Model(inputs=[user_input, movie_input], outputs=output)
        model.compile(optimizer=Adam(lr=0.00001), loss='mean_squared_error')

        return model
    
    def train_model(self, data, epochs=5, batch_size=64, validation_split=0.2):
        # Split the data into training and testing sets
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        # Get the number of unique users and movies
        num_users = data['user'].nunique()
        num_movies = data['movie'].nunique()

        # Create and train the model
        self.model = self.create_model(num_users, num_movies)
        self.model.fit([train_data['user'], train_data['movie']], train_data['rating'],
                       epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def evaluate_model(self, test_data, threshold=0.55):
        # Evaluate the model
        eval_result = self.model.evaluate([test_data['user'], test_data['movie']], test_data['rating'])
        print("Evaluation Result - Loss: {:.4f}".format(eval_result))

        # Predict ratings on the test data
        predictions = self.model.predict([test_data['user'], test_data['movie']])

        # Convert predictions to binary values (0 or 1) based on the threshold
        binary_predictions = (predictions >= threshold).astype(int)

        # Create an "is_liked" column based on the rating threshold
        test_data["is_liked"] = np.where(test_data['rating'] >= 3.5, 1, 0)

        # Calculate accuracy
        accuracy = sum((binary_predictions == test_data['is_liked'].values.reshape(-1, 1)).all(axis=1)) / len(test_data)
        print("Accuracy: {:.2%}".format(accuracy))

    def recommend_movies(self, user_id, model_file=f'{path}/collaborative_filtering_model.h5', top_n=10):
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
        predicted_ratings = model.predict([np.array([encoded_user_id[0]] * len(all_movies)), encoded_movies], verbose=0)

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

    def get_model(self, model_file=f'{path}/collaborative_filtering_model.h5'):
        # Get the model
        return load_model(f'{path}/collaborative_filtering_model.h5')
    
    def save_model(self, model_file=f'{path}collabrative_filtering_model.h5'):
        # Save the model
        self.model.save(model_file)
    
    
