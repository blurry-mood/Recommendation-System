import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from ast import literal_eval
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

credits = pd.read_csv('../input/the-movies-dataset/credits.csv')
keywords = pd.read_csv('../input/the-movies-dataset/keywords.csv')
metadata = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv', low_memory=False)

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata = metadata.drop([19730, 29503, 35587])
metadata['id'] = metadata['id'].astype('int')

metadata = metadata.merge(keywords, on='id')
metadata = metadata.merge(credits, on='id')

metadata = metadata.head(35000)

def get_name(x,jb_idx):
    jobs = ['Producer','Director']
    for i in x:
        if i['job'] == jobs[jb_idx]:
            return i['name']
    return np.nan
    
    
 def preprocess_text(x):
    if isinstance(x, str):
        lower_case = str.lower(x)
        punct = re.sub(r'[^\w\s]', '', lower_case)
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(punct)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        return ' '.join(filtered_sentence)
    else :
        return ''
        
        
        
 def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []
    
    
  def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
            
            
            
  def gather(x):
    return  ' '.join(x['keywords'])+ ' ' + ' '.join(x['cast']) + ' ' + x['director']+ ' ' + x['producer']+ ' ' + ' '.join(x['genres'])
    
    
  def preprocess_gather(metadata):
    variables = ['cast', 'keywords', 'genres']
    for variable in variables:
        metadata[variable] = metadata[variable].apply(literal_eval)
        metadata[variable] = metadata[variable].apply(get_list)
        metadata[variable] = metadata[variable].apply(clean_data)
    variables = ['producer','director']
    metadata['crew'] = metadata['crew'].apply(literal_eval) 
    for i in range(len(variables)):
        metadata[variables[i]] = metadata['crew'].apply(get_name,args=(i,))
        metadata[variables[i]] = metadata[variables[i]].apply(clean_data)
    metadata['overview'] = metadata['overview'].apply(preprocess_text)
    metadata['gathered'] = metadata.apply(gather,axis=1)
    return metadata['gathered']
    
    
data = preprocess_gather(metadata)
count = CountVectorizer(stop_words='english')
embeddings = count.fit_transform(data)
cosine_sim = cosine_similarity(embeddings,embeddings)

def content_based_model(favrt_movies,top_n=10,metadata=metadata):
    recommended_movies = []
    metadata = metadata.reset_index()
    indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
    index = []
    sim_scores = []
    for i in range(len(favrt_movies)):
        index.append(indices[favrt_movies[i]])
        sim_scores.append(list(enumerate(cosine_sim[index[i]])))
    sim_scores = np.array(sim_scores)
    sim_scores = sim_scores.reshape((sim_scores.shape[0]*sim_scores.shape[1],sim_scores.shape[2]))
    sim_scores = list(sim_scores)
    print(len(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = metadata['title'].iloc[movie_indices]
    return recommended_movies
