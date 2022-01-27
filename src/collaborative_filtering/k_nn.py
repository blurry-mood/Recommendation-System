from surprise.model_selection import train_test_split
from surprise import KNNBasic, accuracy
from surprise import Dataset, Reader
from os.path import join, split
import pandas as pd

_HERE = split(__file__)[0]
dataset_path = join(_HERE, '..', '..', 'dataset', 'movielens', 'ratings.csv')

reader = Reader(line_format='user item rating timestamp', sep=',' , rating_scale=(0.5, 5), skip_lines=162541*150)
data = Dataset.load_from_file(dataset_path, reader=reader)
trainset, testset = train_test_split(data, test_size=.25)

sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }
algo = KNNBasic(k=2, sim_options=sim_options, verbose=True)

algo.fit(trainset)
test = algo.test(testset, verbose=True)

accuracy.rmse(test)
