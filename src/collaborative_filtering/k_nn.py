#!/usr/bin/env python
# coding: utf-8

# In[1]:


from surprise.model_selection import train_test_split
from surprise import KNNBasic, accuracy
from surprise import Dataset, Reader
from surprise.dump import dump

from recmetrics import rmse, mse, mark, mark_plot

from os.path import join, split
from random import sample

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import *


# In[2]:


_HERE = '' # split(__file__)[0]


# # Read dataset

# In[ ]:


ratings = ratings_df()
ratings = ratings.query('rating >=3')
ratings = ratings.sample(n=1000)
ratings.reset_index(drop=True, inplace=True)

ratings.head()


# # k-NN Model training

# In[ ]:


reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader=reader)
trainset, testset = train_test_split(data, test_size=.25)


# In[ ]:


sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }
algo = KNNBasic(k=2, sim_options=sim_options, verbose=True)


# In[ ]:


algo.fit(trainset)


# In[ ]:


testset[0]


# In[ ]:





# In[ ]:


preds = algo.test(testset, verbose=False)


# In[ ]:


preds = pd.DataFrame(preds)
preds.drop("details", inplace=True, axis=1)
preds.columns = ['userId', 'movieId', 'actual', 'cf_predictions']
preds.head()


# # Recommendations

# In[ ]:


cf_model = preds.pivot_table(index='userId', columns='movieId', values='cf_predictions').fillna(0)
cf_model.head()


# In[ ]:


test = preds.copy().groupby('userId', as_index=False)['movieId'].agg({'actual': (lambda x: list(set(x)))})
test = test.set_index("userId")
test.head()


# # k-NN recommendations

# In[ ]:


def get_users_predictions(user_id, n, model):
    recommended_items = pd.DataFrame(model.loc[user_id])
    recommended_items.columns = ["predicted_rating"]
    recommended_items = recommended_items.sort_values('predicted_rating', ascending=False)    
    recommended_items = recommended_items.head(n)
    return recommended_items.index.tolist()


# In[ ]:


# make recommendations for all members in the test data
cf_recs = [] = []
for user in tqdm(test.index, desc='k-NN recommendations', total=len(test.index)):
    cf_predictions = get_users_predictions(user, 10, cf_model)
    cf_recs.append(cf_predictions)
        
test['cf_predictions'] = cf_recs
test.head()


# # Popularity-based recommendations

# In[ ]:


#make recommendations for all members in the test data
popularity_recs = ratings.movieId.value_counts().head(10).index.tolist()

pop_recs = []
for user in tqdm(test.index, desc='Popularity-based recommendations', total=len(test.index)):
    pop_predictions = popularity_recs
    pop_recs.append(pop_predictions)
        
test['pop_predictions'] = pop_recs
test.head()


# # Random recommendations

# In[ ]:


# make recommendations for all members in the test data

ran_recs = []
movies = ratings.movieId.values.tolist()

for user in tqdm(test.index, desc='Random recommendations', total=len(test.index)):
    random_predictions = sample(movies, 10)
    ran_recs.append(random_predictions)
        
test['random_predictions'] = ran_recs
test.head()


# In[ ]:





# # Model Evaluation

# In[ ]:


actual = test.actual.values.tolist()
cf_predictions = test.cf_predictions.values.tolist()
pop_predictions = test.pop_predictions.values.tolist()
random_predictions = test.random_predictions.values.tolist()


# In[ ]:


pop_mark = []
for K in np.arange(1, 11):
    pop_mark.extend([mark(actual, pop_predictions, k=K)])
pop_mark


# In[ ]:


random_mark = []
for K in np.arange(1, 11):
    random_mark.extend([mark(actual, random_predictions, k=K)])
random_mark


# In[ ]:


cf_mark = []
for K in np.arange(1, 11):
    cf_mark.extend([mark(actual, cf_predictions, k=K)])
cf_mark


# In[ ]:


test.head()


# In[ ]:


print("MSE: ", mse(preds.actual, preds.cf_predictions))
print("RMSE: ", rmse(preds.actual, preds.cf_predictions))


# In[ ]:


mark_scores = [random_mark, pop_mark, cf_mark]
index = range(1,11)
names = ['Random Recommender', 'Popularity Recommender', 'Collaborative Filter']

fig = plt.figure(figsize=(15, 7))
mark_plot(mark_scores, model_names=names, k_range=index)


# # Saving predictions & model

# In[ ]:


dumping_path = join(_HERE, '..', '..', 'artifacts', 'k-nn.pkl')
dump(dumping_path, algo=algo, verbose=1)


# # Inference stage

# In[ ]:


recommend(['Iron Man', 'Fast and Furious'], 10)


# # Transform notebook to Python script

# In[ ]:


get_ipython().system('jupyter nbconvert --to script k_nn.ipynb')


# In[ ]:





# In[ ]:


collab_model(algo, ['Iron Man (1951)', 'Toy Story (1995)'], 3)


# In[ ]:





# In[ ]:




