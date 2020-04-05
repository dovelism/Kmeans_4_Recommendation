# -*- coding: utf-8 -*-
"""
@Author:Dovelism
@Date:2020/4/5  11:30
@Description: Kmeans'implement for recommendation，using Movielens
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import helper

# Import the Movies dataset
movies = pd.read_csv('movies.csv')
#movies.head()

# Import the ratings dataset
ratings = pd.read_csv('ratings.csv')
#ratings.head()

print('The dataset contains: ', len(ratings), ' ratings of ', len(movies), ' movies.')

genre_ratings = helper.get_genre_ratings(ratings, movies, ['Romance', 'Sci-Fi'], ['avg_romance_rating', 'avg_scifi_rating'])
#genre_ratings.head()

biased_dataset = helper.bias_genre_rating_dataset(genre_ratings, 3.2, 2.5)

print( "Number of records: ", len(biased_dataset))
#biased_dataset.head()



helper.draw_scatterplot(biased_dataset['avg_scifi_rating'],
                        'Avg scifi rating', biased_dataset['avg_romance_rating'],
                        'Avg romance rating')
# turn our dataset into a list
X = biased_dataset[['avg_scifi_rating','avg_romance_rating']].values

# TODO: Import KMeans
from sklearn.cluster import KMeans

# TODO: Create an instance of KMeans to find two clusters
#kmeans_1 = KMeans(n_clusters=2)

# TODO: use fit_predict to cluster the dataset
#predictions = kmeans_1.fit_predict(X)

# Plot
#helper.draw_clusters(biased_dataset, predictions)


# TODO: Create an instance of KMeans to find three clusters
#kmeans_2 = KMeans(n_clusters=3)

# TODO: use fit_predict to cluster the dataset
#predictions_2 = kmeans_2.fit_predict(X)

# Plot
#helper.draw_clusters(biased_dataset, predictions_2)

# Create an instance of KMeans to find four clusters
kmeans_3 = KMeans(n_clusters =4)

# use fit_predict to cluster the dataset
predictions_3 = kmeans_3.fit_predict(X)

# Plot
#helper.draw_clusters(biased_dataset, predictions_3)

# Choose the range of k values to test.
# We added a stride of 5 to improve performance. We don't need to calculate the error for every k value
possible_k_values = range(2, len(X)+1, 5)
#print(possible_k_values)

# Calculate error values for all k values we're interested in
errors_per_k = [helper.clustering_errors(k, X) for k in possible_k_values]

# Look at the values of K vs the silhouette score of running K-means with that value of k
list(zip(possible_k_values, errors_per_k))

# Plot the each value of K vs. the silhouette score at that value
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlabel('K - number of clusters')
ax.set_ylabel('Silhouette Score (higher is better)')
ax.plot(possible_k_values, errors_per_k)

# Ticks and grid
xticks = np.arange(min(possible_k_values), max(possible_k_values)+1, 5.0)
ax.set_xticks(xticks, minor=False)
ax.set_xticks(xticks, minor=True)
ax.xaxis.grid(True, which='both')
yticks = np.arange(round(min(errors_per_k), 2), max(errors_per_k), .05)
ax.set_yticks(yticks, minor=False)
ax.set_yticks(yticks, minor=True)
ax.yaxis.grid(True, which='both')

# Create an instance of KMeans to find seven clusters
kmeans_4 = KMeans(n_clusters=7)

# use fit_predict to cluster the dataset
predictions_4 = kmeans_4.fit_predict(X)

# plot
helper.draw_clusters(biased_dataset, predictions_4, cmap='Accent')


biased_dataset_3_genres = helper.get_genre_ratings(ratings, movies,
                                                     ['Romance', 'Sci-Fi', 'Action'],
                                                     ['avg_romance_rating', 'avg_scifi_rating', 'avg_action_rating'])
biased_dataset_3_genres = helper.bias_genre_rating_dataset(biased_dataset_3_genres, 3.2, 2.5).dropna()

print( "Number of records: ", len(biased_dataset_3_genres))
biased_dataset_3_genres.head()

X_with_action = biased_dataset_3_genres[['avg_scifi_rating',
                                         'avg_romance_rating',
                                         'avg_action_rating']].values
# TODO: Create an instance of KMeans to find seven clusters
kmeans_5 = KMeans(n_clusters=7)

# TODO: use fit_predict to cluster the dataset
predictions_5 = kmeans_5.fit_predict(X_with_action)

# plot
helper.draw_clusters_3d(biased_dataset_3_genres, predictions_5)

# Merge the two tables then pivot so we have Users X Movies dataframe
ratings_title = pd.merge(ratings, movies[['movieId', 'title']], on='movieId' )
user_movie_ratings = pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')

print('dataset dimensions: ', user_movie_ratings.shape, '\n\nSubset example111:')
user_movie_ratings.iloc[:6, :10]

n_movies = 30
n_users = 18
most_rated_movies_users_selection = helper.sort_by_rating_density(user_movie_ratings, n_movies, n_users)

print('dataset dimensions: ', most_rated_movies_users_selection.shape)
most_rated_movies_users_selection.head()

helper.draw_movies_heatmap(most_rated_movies_users_selection)



user_movie_ratings =  pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')
most_rated_movies_1k = helper.get_most_rated_movies(user_movie_ratings, 1000)

sparse_ratings = csr_matrix(pd.SparseDataFrame(most_rated_movies_1k).to_coo())

# 20 clusters
predictions = KMeans(n_clusters=20, algorithm='full').fit_predict(sparse_ratings)

predictions.shape
type(most_rated_movies_1k.reset_index())

max_users = 70
max_movies = 50

clustered = pd.concat([most_rated_movies_1k.reset_index(), pd.DataFrame({'group':predictions})], axis=1)

helper.draw_movie_clusters(clustered, max_users, max_movies)

# Pick a cluster ID from the clusters above
cluster_number = 6

# filter to only see the region of the dataset with the most number of values
n_users = 75
n_movies = 300
cluster = clustered[clustered.group == cluster_number].drop(['index', 'group'], axis=1)

cluster = helper.sort_by_rating_density(cluster, n_movies, n_users)
print(cluster)
helper.draw_movies_heatmap(cluster, axis_labels=False)

cluster.fillna('').head()


# Pick a movie from the table above since we're looking at a subset
movie_name = "American Beauty (1999)"


cluster[movie_name].mean()

# The average rating of 20 movies as rated by the users in the cluster
cluster.mean().head(20)

# TODO: Pick a user ID from the dataset
# Look at the table above outputted by the command "cluster.fillna('').head()"
# and pick one of the user ids (the first column in the table)
user_id = 5
#print(cluster)
# Get all this user's ratings
user_2_ratings  = cluster.loc[user_id, :]

# Which movies did they not rate? (We don't want to recommend movies they've already rated)
user_2_unrated_movies =  user_2_ratings[user_2_ratings.isnull()]
print(user_2_unrated_movies.head())
# What are the ratings of these movies the user did not rate?
avg_ratings = pd.concat([user_2_unrated_movies, cluster.mean()], axis=1, join='inner').loc[:,0]

# Let's sort by rating so the highest rated movies are presented first
avg_ratings.sort_values(ascending=False)[:20]


