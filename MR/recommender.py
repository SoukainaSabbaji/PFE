import pandas as pd 
import numpy as np 

#loading the first dataset containing the movie_id title and cast
ds1 = pd.read_csv('tmdb_5000_credits.csv')
#loadind the second dataset with additional infos
ds2 = pd.read_csv('tmdb_5000_movies.csv')

ds1.columns =['id','title','cast','crew']
#changing the title of the movie id column id so it can 
#match the one in the second dataset
ds2 = ds2.merge(ds1,on='id')
#merging the two

#mean/average of votes 
avg = ds2['vote_average'].mean()

#limiting the number of movies selected
#con here refers to our condition
con = ds2['vote_count'].quantile(0.95)
#movies with a vote count > 95% will be taken from the list

#new list of movies to be used 
movies_list = ds2.copy().loc[ds2['vote_count'] >= con]

def weighted_rating(x, con=con, avg=avg):
    v = x['vote_count']
    R = x['vote_average']
    #this is based on the imdb formula
    #in which (con=1838, avg=6.09)
    return (v/(v+con) * R) + (con/(con+v) * avg)

#defining score and calculating its value using 
#our weighted_rating function 
movies_list['score'] = movies_list.apply(weighted_rating, axis=1)

#sorting the movies
movies_list = movies_list.sort_values('score', ascending=False)
movies_list[['title_x', 'vote_count', 'vote_average', 'score']].head(10)

#finding similarities between the movies
rom sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')
#Replace NaN with an empty string
ds2['overview'] = ds2['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(ds2['overview'])

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(ds2.index, index=ds2['title_x']).drop_duplicates()


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
        
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    #Returning the top 10 most similar movies
    
    
    return ds2['title_x'].iloc[movie_indices]




