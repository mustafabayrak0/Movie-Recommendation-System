import sys
import pandas as pd
import numpy as np
import random
import re
from sklearn.preprocessing import MultiLabelBinarizer
from mlxtend.frequent_patterns import apriori, association_rules

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression as LogReg
from sklearn import preprocessing
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score, confusion_matrix
import os

import re
import datetime

import warnings
warnings.filterwarnings("ignore")

from surprise import Dataset
from surprise import SVD
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Function to select distinct users which are watched most number of movies
def prepare_data(movies, ratings):
    movie_data_ratings_data=movies.merge(ratings,on = 'movieId', how='inner')
    # Group movies_data_ratings_data by userId and create a new column called watched_movies which contains name of all movies watched by the user
    movie_data_ratings_data_grouped_by_userId = movie_data_ratings_data.groupby("userId")
    movie_data_ratings_data_grouped_by_userId = movie_data_ratings_data_grouped_by_userId["title"].apply(list).reset_index(name="watched_movies")
    
    # Get the number of movies watched by each user, split strings ',' and count the number of movies (apply function)
    movie_data_ratings_data_grouped_by_userId["number_of_movies_watched"] = movie_data_ratings_data_grouped_by_userId["watched_movies"].apply(lambda x: len(x))
    # Sort the dataframe by number_of_movies_watched column in descending order
    movie_data_ratings_data_grouped_by_userId = movie_data_ratings_data_grouped_by_userId.sort_values(by="number_of_movies_watched", ascending=False)
    # return userId of top 10000 users
    movie_data_ratings_data_grouped_by_userId = movie_data_ratings_data_grouped_by_userId.iloc[:500]
    top_1000_users = movie_data_ratings_data_grouped_by_userId["userId"]
    # Reset index of movie_data_ratings_data_grouped_by_userId
    movie_data_ratings_data_grouped_by_userId = movie_data_ratings_data_grouped_by_userId.reset_index(drop=True)
    # Filter movie_data_ratings_data by top_1000_users
    movie_data_ratings_data = movie_data_ratings_data[movie_data_ratings_data["userId"].isin(top_1000_users)]
    # Reset index of movie_data_ratings_data
    movie_data_ratings_data = movie_data_ratings_data.reset_index(drop=True)
    # Drop number_of_movies_watched column
    movie_data_ratings_data_grouped_by_userId = movie_data_ratings_data_grouped_by_userId.drop(columns=["number_of_movies_watched"])
    mlb = MultiLabelBinarizer()
    movie_data_ratings_data_encoded = movie_data_ratings_data_grouped_by_userId.join(pd.DataFrame(mlb.fit_transform(movie_data_ratings_data_grouped_by_userId.pop('watched_movies')),
                        columns=mlb.classes_,
                        index=movie_data_ratings_data_grouped_by_userId.index))
    # User Ids column
    user_ids = movie_data_ratings_data_encoded["userId"]
    # Drop userId column
    movie_data_ratings_data_encoded = movie_data_ratings_data_encoded.drop(columns=["userId"])
    # Keep 1000 columns with largest sum
    movie_data_ratings_data_encoded = movie_data_ratings_data_encoded.iloc[:, movie_data_ratings_data_encoded.columns.isin(movie_data_ratings_data_encoded.sum().nlargest(100).index)]
    movie_data_ratings_data_encoded["userId"] = user_ids
    # Set that column as first column
    movie_data_ratings_data_encoded = movie_data_ratings_data_encoded[["userId"] + [col for col in movie_data_ratings_data_encoded.columns if col != "userId"]]
    return movie_data_ratings_data_encoded

# # Function to prepare data for appriori rule
# def prepare_data_apriori(movie_data_ratings_data_encoded):
#     movie_data_ratings_data=movies.merge(ratings,on = 'movieId', how='inner')
#     # Select 1000 random samples
#     movie_data_ratings_data = movie_data_ratings_data.sample(n=1000, random_state=42)
#     # Group movies_data_ratings_data by userId and create a new column called watched_movies which contains name of all movies watched by the user
#     movie_data_ratings_data_grouped_by_userId = movie_data_ratings_data.groupby("userId")
#     movie_data_ratings_data_grouped_by_userId = movie_data_ratings_data_grouped_by_userId["title"].apply(list).reset_index(name="watched_movies")
#     # Create a new dataframe which contains encoded watched_movies column
#     mlb = MultiLabelBinarizer()
#     movie_data_ratings_data_grouped_by_userId = movie_data_ratings_data_grouped_by_userId.iloc[:10000] # to reduce the size of dataframe to 1000 rows
#     movie_data_ratings_data_encoded = movie_data_ratings_data_grouped_by_userId.join(pd.DataFrame(mlb.fit_transform(movie_data_ratings_data_grouped_by_userId.pop('watched_movies')),
#                           columns=mlb.classes_,
#                           index=movie_data_ratings_data_grouped_by_userId.index))
    
#     return movie_data_ratings_data_grouped_by_userId, movie_data_ratings_data_encoded

    

def appriori_rule(movie_data_ratings_data_encoded,flag =False):
    # Code to run appriori rule
    print("Running apriori rule...")
    # min_support = 0.8
    # frequent_itemsets_5 = apriori(movie_data_ratings_data_encoded.drop(columns=["userId"]), min_support=min_support, use_colnames=True, max_len=5)
    # rules_5 = association_rules(frequent_itemsets_5, metric="confidence", min_threshold=0.5)
    # if flag == False:
    #     # Create a dictionary, key is movie names and values is how many times they appear in the rules
    #     movie_names = {}
    #     for index, row in rules_5.iterrows():
    #          for antecedent in row["antecedents"]:
    #              # Split the antecedent string to get the movie name
    #              antecedent = re.split('\'', str(antecedent))[1]
    #              movie_names[antecedent] = movie_names.get(antecedent, 0) + 1
    #          for consequent in row["consequents"]:
    #             movie_names[consequent] = movie_names.get(consequent, 0) + 1
    # return rules_5
    
def kmeans_clustering(movies,ratings,tags):
    # Merge tags and ratings
    tags_ratings = pd.merge(tags, ratings, on=['userId','movieId'], how='outer', suffixes=('_tags', ' '))
    # Drop columns that contain _rating
    tags_ratings = tags_ratings.drop(columns=['timestamp_tags'])
    # Merge tags_ratings and movies
    data = pd.merge(tags_ratings, movies, on=['movieId'], how='outer', suffixes=(' ', '_movie'))
    # Rename timestamp column as timestamp
    data = data.rename(columns={'timestamp ': 'timestamp'})
    # Change column order
    data = data[["movieId","userId", "title", "rating","genres" ,"tag","timestamp"]]
    # Drop rows with NaN timestamp
    data = data.dropna(subset=["timestamp"])
    #Convert timestamp in seconds to datetime format
    data['timestamp'] = data['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
    data["timestamp"] = data['timestamp'].astype('datetime64[ns]')
    
    # Relabel ratings to 1 and 0
    data['rating'] = data['rating'].apply(lambda x: 1 if x > 3 else 0)

    #Delete special characters
    data['tag'] = data['tag'].apply(lambda x: str(x) if x is not np.nan else np.nan)
    data['tag'] = data['tag'].map(lambda x: re.sub(r'([^\s\w]|_)+', '', x) if x is not np.nan else np.nan)
    # Convert to lowercase
    data['tag'] = data['tag'].str.lower()
    
    # Drop rows with NaN tag
    filled_tags = data.dropna(subset=['tag'])
    # Initialize tokenizer
    tokenizer = Tokenizer()
    # Fit on tags
    tokenizer.fit_on_texts(filled_tags['tag'])
    # Create sequences
    sequences = tokenizer.texts_to_sequences(filled_tags['tag'])
    # Find number of unique tokens
    word_index = tokenizer.word_index
    # Pad sequences
    pseq = pad_sequences(sequences)
    # Convert to dataframe
    pdseq = pd.DataFrame(pseq)
    
    # Initialize CountVectorizer
    vectorizer = CountVectorizer(stop_words='english',decode_error='ignore', analyzer='word')
    # Fit and transform corpus
    corpus = filled_tags['tag'].values
    wordvec = vectorizer.fit_transform(corpus.ravel())
    # Convert to array
    wordvec = wordvec.toarray()
    # Get words
    words = vectorizer.get_feature_names_out()
    # Convert to dataframe
    pdwordvec = pd.DataFrame(wordvec,columns=words)
    
    
    # Initialize a dictionary to store the embeddings
    embeddings_index = {}
    # Open the file
    f = open('/home/mustafa/Desktop/Courses/Data-Mining/Project/15/glove.6B.100d.txt')
    # Loop through each line
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    # Close the file
    f.close()
    # Initialize embedding matrix
    embedding_matrix = np.zeros((len(words), 100))
    # Loop through each word
    for i in range(len(words)):
        embedding_vector = embeddings_index.get(words[i])
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Convert to dataframe
    pdembedding = pd.DataFrame(embedding_matrix.T,columns=words)
    
    # Cluster word embeddings data
    kmeans = KMeans(init='k-means++', n_clusters=10)
    # Fit kmeans object to word embeddings data
    kmeans.fit(pdwordvec.T)
    # Get cluster labels
    clusters = kmeans.labels_
    # Create a column to store cluster labels
    # data['cluster'] = clusters
    return clusters

def ask_user_to_rate_movies(movie_data_ratings_data_encoded,movies,ratings,tags):
    
    # Code to ask user to rate movies
    print("Asking user to rate movies...")
    # Get movie names from movie_data_ratings_data_encoded
    movie_names = list(movie_data_ratings_data_encoded.columns[1:])
    
    # Find movie ids of that movies (keep in a dictionary)
    movie_ids = {}
    for movie_name in movie_names:
        movie_ids[movie_name] = movies[movies["title"] == movie_name]["movieId"].iloc[0]
        
    # Create a list to store values of dictionary
    movie_ids_list = list(movie_ids.values())
    # Create a list to store userIds
    user_ids = movie_data_ratings_data_encoded["userId"].tolist()
    # print("Please rate the following movie (1-5):")
    # Filter movies, ratings and tags by movie_ids_list and user_ids
    movies_filtered = movies[movies["movieId"].isin(movie_ids_list)]
    ratings_filtered = ratings[ratings["movieId"].isin(movie_ids_list)]
    tags_filtered = tags[tags["movieId"].isin(movie_ids_list)]
    
    # Select 10 random movies and add them to movies_to_ask list with name and movieId
    movies_to_ask = []
    for i in range(10):
        random_movie = random.choice(movie_names)
        movies_to_ask.append({"name": random_movie, "movieId": movie_ids[random_movie]})
        movie_names.remove(random_movie)
    
    # ratings = ratings[ratings["userId"].isin(user_ids)]
    # tags = tags[tags["userId"].isin(user_ids)]
    
    # movies_to_ask = []
    
    # clustered_df = kmeans_clustering(movies_filtered,ratings_filtered,tags_filtered)
    # print(clustered_df)
    return movies_to_ask, movies_filtered,ratings_filtered,tags_filtered
    
def knn_recommendation():
    # Code to run KNN model
    print("Running KNN recommendation model...")

def collaborative_filtering_recommendation():
    # Code to run Collaborative Filtering model
    print("Running Collaborative Filtering recommendation model...")
    
def svd_recommendation(current_user_id,movies_filtered,ratings_filtered,tags_filtered):
    # Code to run SVD model
    print("Running SVD recommendation model...")
    # Define the rating scale (here, assuming ratings are from 1 to 5)
    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(ratings_filtered[['userId', 'movieId', 'rating']], reader)

    # Train-test split
    trainset, testset = train_test_split(data, test_size=0.2)  # 80-20 split

    # Initialize and train the SVD algorithm
    svd = SVD()
    svd.fit(trainset)

    # Get a list of unique user IDs from your dataset
    user_ids = ratings_filtered['userId'].unique()

    # Select a few sample user IDs for whom you want recommendations
    sample_users = [current_user_id]  # Replace these with actual user IDs from your dataset

    for user_id in sample_users:
        # Generate top N recommendations for the user
        user_recommendations = []
        items_to_ignore = ratings_filtered[ratings_filtered['userId'] == user_id]['movieId'].tolist()
        for movie_id in ratings_filtered['movieId'].unique():
            if movie_id not in items_to_ignore:
                predicted_rating = svd.predict(user_id, movie_id).est
                user_recommendations.append((movie_id, predicted_rating))
    
        # Sort recommendations by predicted rating in descending order
        user_recommendations.sort(key=lambda x: x[1], reverse=True)
        # Display top N recommendations for the user
        print(f"Top recommendations for User {user_id}:")
        for movie_id, predicted_rating in user_recommendations[:10]:
            movie_title = movies_filtered[movies_filtered['movieId'] == movie_id]['title'].values[0]
            print(f"Movie: {movie_title} (Predicted Rating: {predicted_rating})")
        print("\n")
    

def main():
    print("Welcome to the Movie Recommendation System!")
    print("Select a model to run:")
    print("1. KNN")
    print("2. Collaborative Filtering")
    print("3. Appriori Rule")
    print("0. Exit")
    
    # Read datasets
    movies = pd.read_csv("/home/mustafa/Desktop/Courses/Data-Mining/Project/ml-25m/movies.csv")
    ratings = pd.read_csv("/home/mustafa/Desktop/Courses/Data-Mining/Project/ml-25m/ratings.csv", sep=",")
    links = pd.read_csv("/home/mustafa/Desktop/Courses/Data-Mining/Project/ml-25m/links.csv", sep=",")
    tags = pd.read_csv("/home/mustafa/Desktop/Courses/Data-Mining/Project/ml-25m/tags.csv", sep=",")
    
    # Set current user's id as largest user id + 1
    current_user_id = ratings["userId"].max() + 1
    
    # Ask user to rate 10 movies, select these movies by using appriori rule
    movie_data_ratings_data_encoded = prepare_data(movies, ratings)
    movies_to_ask,movies_filtered,ratings_filtered,tags_filtered = ask_user_to_rate_movies(movie_data_ratings_data_encoded,movies,ratings,tags)
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Please rate the following movie (1-10):")
    print("If you haven't watched the movie, please enter 0.")
    for i in range(10):
        print(movies_to_ask[i]["name"])
        # rating = input("Enter your rating: ")
        # Check if rating is valid
        rating = "5"
        while rating not in [str(i) for i in range(11)]:
            rating = input("Invalid rating. Please enter a valid rating: ")
        rating = "5"
        movies_to_ask[i]["rating"] = str(int(rating) / 2)
        movies_to_ask[i]["userId"] = current_user_id
        # Add current timestamp
        movies_to_ask[i]["timestamp"] = datetime.datetime.now().timestamp()
        
    # Drop name column
    movies_to_ask = pd.DataFrame(movies_to_ask).drop(columns=["name"])
    # Add these ratings to ratings dataframe
    ratings = pd.concat([ratings, movies_to_ask], ignore_index=True)
    
    print("Please rate the following movie (1-10):")
    # Add these ratings to ratings dataframe
    
    
    
    while True:
        # choice = input("Enter your choice: ")
        choice = "4"

        if choice == "1":
            knn_recommendation()
        elif choice == "2":
            collaborative_filtering_recommendation()
        elif choice == "3":
            # prepare_data(movies, ratings)
            rules = appriori_rule(movie_data_ratings_data_encoded)
        elif choice == "4":
            # SVD
            svd_recommendation(current_user_id,movies_filtered,ratings_filtered,tags_filtered)


            
        elif choice == "0":
            print("Exiting...")
            sys.exit()
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()
    
# Accuracylerine göre weighted sum yapıp suggestion yap
# Suggestionların detaylarını görmek isteyenlere her modelin ne önerdiğini göster
