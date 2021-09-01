import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Loading the files
movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")
a = movies.head()
# print(a)


# Spliting the files into a list, for better usage
movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))
b = movies.head()
# print(b)


from collections import Counter

# Counting how many gender labels there is
genres_counts = Counter(g for genres in movies['genres'] for g in genres)
# print(f"There are {len(genres_counts)} genre labels.")
c = genres_counts
# print(c)


# Removing the 'no genre listed' label
movies = movies[movies['genres']!='(no genres listed)']
del genres_counts['(no genres listed)']

d = genres_counts
# print(d)

# Printing the most common 5
# print("The 5 most common genres: \n", genres_counts.most_common(5))


# Visualizing the genre popularity with a barplot

genres_counts_df = pd.DataFrame([genres_counts]).T.reset_index()
genres_counts_df.columns = ['genres', 'count']
genres_counts_df = genres_counts_df.sort_values(by='count', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='genres', y='count', data=genres_counts_df, palette='viridis')
plt.xticks(rotation=90)
# plt.show()


# Separating the year from the movie title
import re

def extract_year_from_title(title):
    t = title.split(' ')
    year = None
    if re.search(r'\(\d+\)', t[-1]):
        year = t[-1].strip('()')
        year = int(year)
    return year

# Testing the function
# title = "Toy Story (1995)"
# year = extract_year_from_title(title)
# print(f"Year of release: {year}")
# print(type(year))


# Applying to all the dataframe
movies['year'] = movies['title'].apply(extract_year_from_title)
e = movies.head()
# print(e)


# Getting the quantity of years covered in the dataset
f = movies['year'].nunique()
# print(f)


# Removing all the movies with null year
# print(f"Original number of movies: {movies['movieId'].nunique()}")

movies = movies[~movies['year'].isnull()]
# print(f"Number of movies after removing null years: {movies['movieId'].nunique()}")

# Dividing into decades
x = 1995

def get_decade(year):
    year = str(year)
    decade_prefix = year[0:3] # get first 3 digits of the year
    decade = f'{decade_prefix}0' # append 0 at the end
    return int(decade)

# g = get_decade(x)
# print(g)

# Another way to get the decade

def round_down(year):
    return year - (year%10)

h = round_down(x)
# print(h)


# Getting all the decades from our dataset
movies['decade'] = movies['year'].apply(round_down)
plt.figure(figsize=(10, 6))
sns.countplot(movies['decade'], palette='Blues')
plt.xticks(rotation=90)
# plt.show()


# Transforming the data
genres = list(genres_counts.keys())

for g in genres:
    movies[g] = movies['genres'].transform(lambda x: int(g in x))


i = movies[genres].head()
# print(i)


# Giving each decade it's own column
movie_decades = pd.get_dummies(movies['decade'])
j = movie_decades.head()
# print(j)


# combining our genres and decades columns
movie_features = pd.concat([movies[genres], movie_decades], axis=1)
k = movie_features.head()
# print(k)


# Building the recommender using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(movie_features, movie_features)
# print(f"Dimensions of our movie features cosine similarity matrix: {cosine_sim.shape}")


# Creating the movie finder function
from fuzzywuzzy import process

def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0]

title = movie_finder('jumanji')
# print(title)

movie_idx = dict(zip(movies['title'], list(movies.index)))
idx = movie_idx[title]
# print(idx)


# Gettint the top 10 most similar movies to Jumanji
n_recommendations = 10
sim_scores = list(enumerate(cosine_sim[idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
sim_scores = sim_scores[1:(n_recommendations+1)]
similar_movies = [i[0] for i in sim_scores]

# print(f"Because you watched {title}")
# print(f"You should also like: \n{movies['title'].iloc[similar_movies]}")


# Testing more movies
def get_content_based_recommendations(title_string, n_recommendations=10):
    title = movie_finder(title_string)
    idx = movie_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    similar_movies = [i[0] for i in sim_scores]
    print(f"Recommendations for {title}")
    print(movies['title'].iloc[similar_movies])

get_content_based_recommendations('aladin', 5)
