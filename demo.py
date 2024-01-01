# Import libraries
import sys
import pandas as pd
import numpy as np
import random
import re
from sklearn.preprocessing import MultiLabelBinarizer
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from keras.preprocessing.text import Tokenizer
# from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import pad_sequences
import os
import datetime
import warnings
warnings.filterwarnings("ignore")
from surprise import Dataset
from surprise import SVD
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import time
from colab_filtering import CollaborativeFilteringModel

path = "/home/mustafa/Desktop/Courses/Data-Mining/Project/15"

##################### DATA PREPROCESSING #####################

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
    movie_data_ratings_data_grouped_by_userId = movie_data_ratings_data_grouped_by_userId.iloc[:1000]
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
    movie_data_ratings_data_encoded = movie_data_ratings_data_encoded.iloc[:, movie_data_ratings_data_encoded.columns.isin(movie_data_ratings_data_encoded.sum().nlargest(1000).index)]
    movie_data_ratings_data_encoded["userId"] = user_ids
    # Set that column as first column
    movie_data_ratings_data_encoded = movie_data_ratings_data_encoded[["userId"] + [col for col in movie_data_ratings_data_encoded.columns if col != "userId"]]
    return movie_data_ratings_data_encoded

def ask_user_to_rate_movies(movie_data_ratings_data_encoded,movies,ratings,tags):
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
    
    # Filter ratings by user_ids
    ratings_filtered = ratings_filtered[ratings_filtered["userId"].isin(user_ids)]
    # Filter tags by user_ids
    tags_filtered = tags_filtered[tags_filtered["userId"].isin(user_ids)]
    
    # Reset index
    movies_filtered = movies_filtered.reset_index(drop=True)
    ratings_filtered = ratings_filtered.reset_index(drop=True)
    tags_filtered = tags_filtered.reset_index(drop=True)
    
    # Call social network function
    betweenness_centrality_df = social_network(movies_filtered, ratings_filtered)
    
    # Select 10 random movies from betweenness_centrality_df
    movie_names_new = betweenness_centrality_df["title"].tolist()
    
    # Select 10 random movies and add them to movies_to_ask list with name and movieId
    movies_to_ask = []
    for i in range(10):
        random_movie = random.choice(movie_names_new)
        movies_to_ask.append({"name": random_movie, "movieId": movie_ids[random_movie]})
        movie_names_new.remove(random_movie)
    
    ratings = ratings[ratings["userId"].isin(user_ids)]
    tags = tags[tags["userId"].isin(user_ids)]
    
    # clustered_df = kmeans_clustering(movies_filtered,ratings_filtered,tags_filtered)
    # print(clustered_df)
    return movies_to_ask, movies_filtered,ratings_filtered,tags_filtered

# Function to prepare data for appriori rule
def prepare_data_apriori(movies,ratings):
    movie_data_ratings_data=movies.merge(ratings,on = 'movieId', how='inner')
    
    
    # Create a dataframe that columns are movie names and rows are userIds and values are ratings
    movie_data_ratings_data = movie_data_ratings_data.pivot(index="userId", columns="title", values="rating")
    # Fill NaN values with 0
    movie_data_ratings_data = movie_data_ratings_data.fillna(0)
    # Reset index
    movie_data_ratings_data = movie_data_ratings_data.reset_index()
    # Drop userId column
    movie_data_ratings_data = movie_data_ratings_data.drop(columns=["userId"])
    # If rating is greater than 1, set it as 1 (liked), else set it as 0 (disliked)
    movie_data_ratings_data = movie_data_ratings_data.applymap(lambda x: 1 if x >= 1 else 0)
    return movie_data_ratings_data


##################### SOCIAL NETWORK #####################

def social_network(movies_df, ratings_df):
    # Create a dictionary to map movie IDs to their names
    movie_names = dict(zip(movies_df.movieId, movies_df.title))
    # Create an empty graph
    G = nx.Graph()

    # Add nodes for users and movies
    for _, row in ratings_df.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        rating = row['rating']
        G.add_node(user_id, type='user')
        G.add_node(movie_id, type='movie')
        G.add_edge(user_id, movie_id, rating=rating)
    movie_degrees = {n: d for n, d in G.degree() if G.nodes[n].get('type') == 'movie'}
    # Number of top movies you want to find
    n = 5 

    # Sort the movies by degree, in descending order
    sorted_movies = sorted(movie_degrees.items(), key=lambda item: item[1], reverse=True)

    # Get the top n movies
    top_n_movies = sorted_movies[:n]

    # Retrieve the names of these top n movies
    top_n_movie_names = [(movie_names.get(movie_id, "Unknown Movie"), degree) for movie_id, degree in top_n_movies]
    
    # Calculate centralities
    betweenness_centrality = nx.betweenness_centrality(G)
    # eigenvector_centrality = nx.eigenvector_centrality(G)
    # closeness_centrality = nx.closeness_centrality(G)
    # degree_centrality = nx.degree_centrality(G)
    
    # Set values in dataframe
    movies_df['betweenness_centrality'] = movies_df['movieId'].map(betweenness_centrality)
    # movies_df['eigenvector_centrality'] = movies_df['movieId'].map(eigenvector_centrality)
    # movies_df['closeness_centrality'] = movies_df['movieId'].map(closeness_centrality)
    # movies_df['degree_centrality'] = movies_df['movieId'].map(degree_centrality)
    
    # Sort dataframe by betweenness_centrality
    movies_df = movies_df.sort_values(by=['betweenness_centrality'], ascending=False)
    # Keep first 50 rows
    movies_df = movies_df.iloc[:50]
    return movies_df

###################### APRIOIRI RULE ######################

def appriori_rule(movie_data_ratings_data_encoded,liked_movies):
    # Code to run appriori rule
    print("Running apriori rule...")
    min_support = 0.3
    frequent_itemsets_5 = apriori(movie_data_ratings_data_encoded, min_support=min_support, use_colnames=True, max_len=5)
    rules_5 = association_rules(frequent_itemsets_5, metric="confidence", min_threshold=0.1)
    # Get number of movies in antecedents that are in liked_movies
    rules_5["liked_movies"] = rules_5["antecedents"].apply(lambda x: len(set(x).intersection(set(liked_movies))))
    # If all liked_movies are zero, print message
    if rules_5["liked_movies"].sum() == 0:
        print("No rules found.")
        return
    # Sort dataframe by liked_movies
    rules_5 = rules_5.sort_values(by=["liked_movies"], ascending=False)
    # Keep first 10 rows
    rules_5 = rules_5.iloc[:10]
    # Drop if consequents are in the liked_movies
    rules_5 = rules_5[~rules_5["consequents"].isin(liked_movies)]
    # Check if liked_movies greater than 0
    if rules_5["liked_movies"].sum() == 0:
        print("No rules found.")
        return
    # Drop rows with less than 1 liked_movies
    rules_5 = rules_5[rules_5["liked_movies"] > 0]
    # Return names of unique movies in consequents
    movies_appriori = rules_5["consequents"].unique()
    return movies_appriori
    
    
###################### K-Means ######################
    
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
    f = open('{path}/glove.6B.100d.txt')
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


###################### KNN ######################
def knn_recommendation(user_id,merged_dataset):
    # Select a random user
    # user_id = np.random.choice(merged_dataset['user id'].unique())

    refined_dataset = merged_dataset.groupby(by=['user id','movie title'], as_index=False).agg({"rating":"mean"})
    user_to_movie_df = refined_dataset.pivot(
        index='user id',
        columns='movie title',
        values='rating').fillna(0)
    user_to_movie_sparse_df = csr_matrix(user_to_movie_df.values)
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(user_to_movie_sparse_df)
    
    def get_similar_users(user, n = 5):
        
        # Seşect a random user
        user = np.random.choice(user_to_movie_df.index)
        # Get the line of the user by user id
        user_line = user_to_movie_df[user_to_movie_df.index == user].values
        knn_input = np.asarray(user_line)
        
        # knn_input = np.asarray([user_to_movie_df.values[user-1]])
    
        distances, indices = knn_model.kneighbors(knn_input, n_neighbors=n+1)
    
        # print("Top",n,"users who are very much similar to the User-",user, "are: ")
        # print(" ")

        # for i in range(1,len(distances[0])):
        #     print(i,". User:", indices[0][i]+1, "separated by distance of",distances[0][i])
        # print("")
        return indices.flatten()[1:] + 1, distances.flatten()[1:]    
    
    
    similar_user_list, distance_list = get_similar_users(user_id,3)
    weightage_list = distance_list/np.sum(distance_list)
    mov_rtngs_sim_users = user_to_movie_df.values[similar_user_list]
    movies_list = user_to_movie_df.columns
    weightage_list = weightage_list[:,np.newaxis] + np.zeros(len(movies_list))
    new_rating_matrix = weightage_list*mov_rtngs_sim_users
    mean_rating_list = new_rating_matrix.sum(axis =0)

    def filtered_movie_recommendations(n = 10):
        zero_indices = np.where(mean_rating_list == 0)[0]
        if len(zero_indices) > 0:
            first_zero_index = zero_indices[-1]
            sortd_index = np.argsort(mean_rating_list)[::-1]
            sortd_index = sortd_index[:list(sortd_index).index(first_zero_index)]
        else:
            sortd_index = np.argsort(mean_rating_list)[::-1]    
        n = min(len(sortd_index),n)
        movies_watched = list(refined_dataset[refined_dataset['user id'] == user_id]['movie title'])
        filtered_movie_list = list(movies_list[sortd_index])
        count = 0
        final_movie_list = []
        for i in filtered_movie_list:
            if i not in movies_watched:
                count+=1
                final_movie_list.append(i)
            if count == n:
                break
        if count == 0:
            print("There are no movies left which are not seen by the input users and seen by similar users. May be increasing the number of similar users who are to be considered may give a chance of suggesting an unseen good movie.")
        else:
            for i in range(len(final_movie_list)):
                print(f"{i+1}.{final_movie_list[i]}")
                
    print("")
    print("Movies recommended based on similar users are: ")
    print("")
    filtered_movie_recommendations(10)

###################### Collaborative Filtering ######################
def collaborative_filtering_recommendation():
    # Code to run Collaborative Filtering model
    print("Running Collaborative Filtering recommendation model...")
    
###################### SVD ######################
    
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
        i = 1
        for movie_id, predicted_rating in user_recommendations[:10]:
            movie_title = movies_filtered[movies_filtered['movieId'] == movie_id]['title'].values[0]
            print(f"{i}.{movie_title}")
            i += 1

def main():
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Welcome to the Movie Recommendation System!")
    # Print wait while loading datasets
    print("Please wait while loading datasets...")
    
    # Read datasets
    movies = pd.read_csv(f"{path}/ml-latest-small/movies.csv")
    ratings = pd.read_csv(f"{path}/ml-latest-small/ratings.csv", sep=",")
    links = pd.read_csv(f"{path}/ml-latest-small/links.csv", sep=",")
    tags = pd.read_csv(f"{path}/ml-latest-small/tags.csv", sep=",")
    
    # Set current user's id as largest user id + 1
    current_user_id = ratings["userId"].max() + 1
    
    # Ask user to rate 10 movies, select these movies by using appriori rule
    movie_data_ratings_data_encoded = prepare_data(movies, ratings)
    movies_to_ask,movies_filtered,ratings_filtered,tags_filtered = ask_user_to_rate_movies(movie_data_ratings_data_encoded,movies,ratings,tags)
    movies_filtered = movies_filtered.drop(columns=["betweenness_centrality"])
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    # Call social network function
    print("Please rate the following movie (1-10):")
    print("If you haven't watched the movie, please enter 0.")
    for i in range(10):
        print(movies_to_ask[i]["name"])
        rating = input("Enter your rating: ")
        # Check if rating is valid
        # rating = "5"
        while rating not in [str(i) for i in range(11)]:
            rating = input("Invalid rating. Please enter a valid rating: ")
        movies_to_ask[i]["rating"] = int(rating) / 2
        movies_to_ask[i]["userId"] = current_user_id
        # Add current timestamp
        movies_to_ask[i]["timestamp"] = datetime.datetime.now().timestamp()
        
    # Get liked movies (rating > 3) and store their name in a list
    liked_movies = []
    for movie in movies_to_ask:
        if float(movie["rating"]) > 3:
            liked_movies.append(movie["name"])
    # Drop name column
    movies_asked = pd.DataFrame(movies_to_ask).drop(columns=["name"])
    
    # Add these ratings to ratings dataframe
    ratings_filtered = pd.concat([ratings_filtered, movies_asked], ignore_index=True)
    # Reset index
    ratings_filtered = ratings_filtered.reset_index(drop=True)
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Select a model to run:")
    print("1. KNN")
    print("2. Collaborative Filtering")
    print("3. Appriori Rule")
    print("4. Singular Value Decomposition")
    print("0. Exit")
    while True:
        choice = input("Enter your choice: ")
        if choice == "1":
            # Add a new column to ratings_filtered dataframe called movie title
            ratings_filtered_knn = ratings_filtered.merge(movies_filtered[["movieId", "title"]], on="movieId", how="inner")
            # Rename columns
            ratings_filtered_knn = ratings_filtered_knn.rename(columns={"userId": "user id", "movieId": "movie id", "title": "movie title"})
            knn_recommendation(current_user_id,ratings_filtered_knn)
        elif choice == "2":
            # # Train and evaluate the model
            # rec_system = CollaborativeFilteringModel(ratings_filtered, movies_filtered)
            # data = rec_system.load_data()
            # rec_system.train_model(data)
            # rec_system.evaluate_model(data)
            # # Save the model
            # rec_system.save_model()
            # print("hi")
            rec_system = CollaborativeFilteringModel(ratings_filtered, movies_filtered)
            recomendations = rec_system.recommend_movies(user_id=current_user_id)
            # Print each movie
            for i, movie in enumerate(recomendations["title"]):
                print(f"{i+1}. {movie}")
        elif choice == "3":
            apriori_encoded_data = prepare_data_apriori(movies_filtered,ratings_filtered)
            movies_appriori = appriori_rule(apriori_encoded_data,liked_movies)
            if movies_appriori is not None:
                movies_list = [list(movie)[0] for movie in movies_appriori]
                for i, movie in enumerate(movies_list):
                    print(f"{i+1}. {movie}")
        elif choice == "4":
            svd_recommendation(current_user_id,movies_filtered,ratings_filtered,tags_filtered)

            
        elif choice == "0":
            print("Exiting...")
            sys.exit()
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()    
