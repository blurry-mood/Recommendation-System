from os.path import join, split
import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNBasic


from fuzzywuzzy import fuzz



def get_movie_index(movie_name):
    return _movies.loc[_movies['title'] == movie_name, 'movieId'].values.tolist()[0]


def get_movie_name(movie_id):
    return _movies.loc[_movies['movieId'] == int(movie_id), 'title'].values.tolist()[0]


def ratings_df():
    return _ratings[['userId', 'movieId', 'rating']]

def get_movie_names():
    return list(set(_movies['title'].values.tolist()))

def get_movie_ids():
    return list(set(_movies['movieId'].values.tolist()))

def fuzzy_matching(movie_names, movie, verbose=False):
    """
    return the closest match via fuzzy ratio. 
    
    Parameters
    ----------    
    mapper: dict, map movie title name to index of the movie in data

    fav_movie: str, name of user input movie
    
    verbose: bool, print log if True

    Return
    ------
    the closest match
    """
    match_tuple = []
    # get match
    for title in movie_names:
        ratio = fuzz.ratio(title.lower(), movie.lower())
        if ratio >= 60:
            match_tuple.append((title, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[-1])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][0]




def get_users_predictions(user_id, n, model):
    recommended_items = pd.DataFrame(model.loc[user_id])
    recommended_items.columns = ["predicted_rating"]
    recommended_items = recommended_items.sort_values('predicted_rating', ascending=False)    
    recommended_items = recommended_items.head(n)
    return recommended_items.index.tolist()

# recommend for a new user
def recommend(fav_movies, topK=10):
    fav_movies = ['Iron Man']
    fav_movie_ids = [get_movie_index(fuzzy_matching(_movie_names, movie)) for movie in fav_movies]
    user = [['0', str(movie_id), 5.0, -1] for movie_id in fav_movie_ids]

    new_ratings = user + _ratings.sample(n=1000).values.tolist()
    new_ratings = pd.DataFrame(new_ratings, columns=['userId', 'movieId', 'rating', 'timestamp'])

    data = Dataset.load_from_df(new_ratings[['userId', 'movieId', 'rating']], reader=Reader(rating_scale=(0, 5))).build_full_trainset()

    sim_options = {'name': 'cosine',
                   'user_based': True  # compute  similarities between users
                   }
    algo = KNNBasic(k=2, sim_options=sim_options, verbose=True)

    algo = algo.fit(data)
    
    movie_ratings = { str(idd): 0 for idd in _movie_ids}
    for _, mv, rating, _ in user:
        movie_ratings[mv] = rating

    new_user = [('0', mv, rating) for mv, rating in movie_ratings.items()]
    
    preds = algo.test(new_user, verbose=False)
    preds = pd.DataFrame(preds)
    preds.drop("details", inplace=True, axis=1)
    preds.columns = ['userId', 'movieId', 'actual', 'cf_predictions']

    cf_model = preds.pivot_table(index='userId', columns='movieId', values='cf_predictions').fillna(0)
    
    recommendations = get_users_predictions('0', topK, cf_model)
    recommendations = [get_movie_name(mv) for mv in recommendations]
    
    return recommendations
    
    
    
# if  True:
# if __name__=='__main__':
_HERE = split(__file__)[0]
data_dir = join(_HERE, '..', '..', 'dataset', 'movielens')

_ratings = pd.read_csv(join(data_dir, 'ratings.csv'))
_tags = pd.read_csv(join(data_dir, 'tags.csv'))
_movies = pd.read_csv(join(data_dir, 'movies.csv'))



_movie_names = get_movie_names()
_movie_ids = get_movie_ids()
# Surprise Full Dataset
_full_dataset = Dataset.load_from_df(ratings_df(), Reader(rating_scale=(0, 5))).build_full_trainset()


def prediction_item(model, item_id):
    predictions = []
    for ui in _full_dataset.all_users():
        predictions.append(model.predict(iid=item_id,uid=ui, verbose = False))
    return predictions

def pred_movies(model, movie_list):
    # Store the id of users
    id_store=[]
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        predictions = prediction_item(model, item_id = i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:10]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store

def collab_model(model, movie_list, top_n=10):
    indices = pd.Series(_movies['title'])
    movie_ids = pred_movies(model, movie_list)
    df_init_users = _ratings[_ratings['userId']==movie_ids[0]]
    for i in movie_ids :
        df_init_users=df_init_users.append(_ratings[_ratings['userId']==i])
        
    # Getting the cosine similarity matrix
    cosine_sim = cosine_similarity(np.array(df_init_users), np.array(df_init_users))
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]
    
    # Creating a Series with the similarity scores in descending order
    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]
    
    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
    
     # Appending the names of movies
    listings = score_series_1.append(score_series_1).append(score_series_3).sort_values(ascending = False)
    recommended_movies = []
    
    # Choose top 50
    top_50_indexes = list(listings.iloc[1:100].index)
    
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies_df['title'])[i])

    return recommended_movies