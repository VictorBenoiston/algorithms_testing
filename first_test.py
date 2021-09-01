import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ratings = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv")
# a = ratings.head()
# print(a)  # 1

movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")
# b = movies.head()
# print(b)  # 2

n_ratings = len(ratings)
n_movies = ratings['movieId'].nunique()
n_users = ratings['userId'].nunique()

# print(f'Numver of ratings: {n_ratings}')
# print(f"Numver of unique movieId's: {n_movies}")
# print(f"Number of unique users: {n_users}")
# print(f"Average number of ratings per user: {round(n_ratings/n_users, 2)}")
# print(f"Average number of ratings per movie: {round(n_ratings/n_movies, 2)}")   # 3

user_freq = ratings[['userId', 'movieId']].groupby('userId').count().reset_index()
user_freq.columns = ['userId', 'n_ratings']
# c = user_freq.head()
# print(c)  # 4

# print(f"Mean number of ratings for a giver user: {user_freq['n_ratings'].mean():.2f}")   # 5


# sns.set_style('whitegrid')
# plt.figure(figsize=(14,5))
# plt.subplot(1, 2, 1)
# ax = sns.countplot(x = "rating", data=ratings, palette='viridis')
# plt.title("Distribution of movie ratings")
#
# plt.subplot(1, 2, 2)
# ax = sns.kdeplot(user_freq['n_ratings'], shade=True, legend=False)
# plt.axvline(user_freq['n_ratings'].mean(), color="k", linestyle="--")
# plt.xlabel("# ratings per user")
# plt.ylabel("density")
# plt.title("Number of movies rated per user")
# plt.show()  # 6


mean_rating = ratings.groupby('movieId')[['rating']].mean()
#
# lowest_rated = mean_rating['rating'].idxmin()
# d = movies.loc[movies['movieId'] == lowest_rated]
# print(d) # 7

highest_rated = mean_rating['rating'].idxmax()
e = movies.loc[movies['movieId'] == highest_rated]
# print(e)  # 8

# f = ratings[ratings['movieId'] == highest_rated]
# print(f)  # 9

movie_stats = ratings.groupby('movieId')[['rating']].agg(['count', 'mean'])
movie_stats.columns = movie_stats.columns.droplevel()

C = movie_stats['count'].mean()
m = movie_stats['mean'].mean()

def bayesian_avg(ratings):
    bayesian_avg = ((C*m+ratings.sum()))/(C+ratings.count())
    return bayesian_avg

bayesian_avg_ratings = ratings.groupby('movieId')['rating'].agg(bayesian_avg).reset_index()
bayesian_avg_ratings.columns = ['movieId', 'bayesian_avg']
movie_stats = movie_stats.merge(bayesian_avg_ratings, on='movieId')

movie_stats = movie_stats.merge(movies[['movieId', 'title']])
g = movie_stats.sort_values('bayesian_avg', ascending=False).head()
# print(g)  # 10

h = movie_stats.sort_values('bayesian_avg', ascending=True).head()
# print(h)  # 11


from scipy.sparse import csr_matrix

def create_X(df):
    """
    Generates a sparse matrix from ratings dataframe.

    Args:
        df: pandas dataframe
    Returns:
        X: sparse matrix
        user_mapper: dict that maps id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that maps movie indices to movie id's
    """
    N = df['userId'].nunique()
    M = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))

    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)

sparsity = X.count_nonzero()/(X.shape[0]*X.shape[1])
# print(f"Matrix sparsity: {round(sparsity*100, 2)}%")  # 12

# from scipy.sparse import save_npz
# save_npz('Users\Dalvani\Desktop\TCC', X)  # 13


from sklearn.neighbors import NearestNeighbors

def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    """
    Finds k-nearest neighbours for a given movie id
    Args:
        movie_id: id of the movie of interest
        X: user-item utility matrix
        k: number of similar movies to retrieve
        metric: distance metric for kNN calculations
    Returns:
        list of k similar movie ID's
    """
    neighbours_ids = []

    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(0, k):
        n = neighbour.item(i)
        neighbours_ids.append(movie_inv_mapper[n])
    neighbours_ids.pop(0)
    return neighbours_ids

movie_titles = dict(zip(movies['movieId'], movies['title']))

movie_id = 2

similar_ids = find_similar_movies(movie_id, X, k=10)
movie_title = movie_titles[movie_id]

print(f"Because you watched {movie_title}")
print('You should also watch: ')
print('-' * 30)
for i in similar_ids:
    print(movie_titles[i])
