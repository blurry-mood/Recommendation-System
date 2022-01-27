#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[5]:


_HERE = ''
dataset_path = join(_HERE, '..', '..', 'dataset', 'movielens', 'ratings.csv')
dataset_path


# # Read dataset

# In[6]:


ratings = pd.read_csv(dataset_path)
ratings = ratings.query('rating >=3')
ratings.reset_index(drop=True, inplace=True)

ratings.head()


# # k-NN Model training

# In[7]:


reader = Reader(line_format='user item rating timestamp', sep=',' , rating_scale=(0.5, 5), skip_lines=162541*150)
data = Dataset.load_from_file(dataset_path, reader=reader)
trainset, testset = train_test_split(data, test_size=.25)


# In[8]:


sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }
algo = KNNBasic(k=2, sim_options=sim_options, verbose=True)


# In[9]:


algo.fit(trainset)


# In[10]:


preds = algo.test(testset, verbose=False)


# In[11]:


preds = pd.DataFrame(preds)
preds.drop("details", inplace=True, axis=1)
preds.columns = ['userId', 'movieId', 'actual', 'cf_predictions']
preds.head()


# # Recommendations

# In[12]:


cf_model = preds.pivot_table(index='userId', columns='movieId', values='cf_predictions').fillna(0)
cf_model.head()


# In[13]:


test = preds.copy().groupby('userId', as_index=False)['movieId'].agg({'actual': (lambda x: list(set(x)))})
test = test.set_index("userId")
test.head()


# # k-NN recommendations

# In[14]:


def get_users_predictions(user_id, n, model):
    recommended_items = pd.DataFrame(model.loc[user_id])
    recommended_items.columns = ["predicted_rating"]
    recommended_items = recommended_items.sort_values('predicted_rating', ascending=False)    
    recommended_items = recommended_items.head(n)
    return recommended_items.index.tolist()


# In[15]:


# make recommendations for all members in the test data
cf_recs = [] = []
for user in tqdm(test.index, desc='k-NN recommendations', total=len(test.index)):
    cf_predictions = get_users_predictions(user, 10, cf_model)
    cf_recs.append(cf_predictions)
        
test['cf_predictions'] = cf_recs
test.head()


# # Popularity-based recommendations

# In[16]:


#make recommendations for all members in the test data
popularity_recs = ratings.movieId.value_counts().head(10).index.tolist()

pop_recs = []
for user in tqdm(test.index, desc='Popularity-based recommendations', total=len(test.index)):
    pop_predictions = popularity_recs
    pop_recs.append(pop_predictions)
        
test['pop_predictions'] = pop_recs
test.head()


# # Random recommendations

# In[18]:


# make recommendations for all members in the test data

ran_recs = []
movies = ratings.movieId.values.tolist()

for user in tqdm(test.index, desc='Random recommendations', total=len(test.index)):
    random_predictions = sample(movies, 10)
    ran_recs.append(random_predictions)
        
test['random_predictions'] = ran_recs
test.head()


# In[25]:





# # Model Evaluation

# In[19]:


actual = test.actual.values.tolist()
cf_predictions = test.cf_predictions.values.tolist()
pop_predictions = test.pop_predictions.values.tolist()
random_predictions = test.random_predictions.values.tolist()


# In[20]:


pop_mark = []
for K in np.arange(1, 11):
    pop_mark.extend([mark(actual, pop_predictions, k=K)])
pop_mark


# In[21]:


random_mark = []
for K in np.arange(1, 11):
    random_mark.extend([mark(actual, random_predictions, k=K)])
random_mark


# In[22]:


cf_mark = []
for K in np.arange(1, 11):
    cf_mark.extend([mark(actual, cf_predictions, k=K)])
cf_mark


# In[23]:


test.head()


# In[24]:


print("MSE: ", mse(preds.actual, preds.cf_predictions))
print("RMSE: ", rmse(preds.actual, preds.cf_predictions))


# In[25]:


mark_scores = [random_mark, pop_mark, cf_mark]
index = range(1,11)
names = ['Random Recommender', 'Popularity Recommender', 'Collaborative Filter']

fig = plt.figure(figsize=(15, 7))
mark_plot(mark_scores, model_names=names, k_range=index)


# # Saving predictions & model

# In[26]:


dumping_path = join(_HERE, '..', '..', 'artifacts', 'k-nn.dump')
dump(dumping_path, algo=algo, verbose=1)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script k_nn.ipynb')

