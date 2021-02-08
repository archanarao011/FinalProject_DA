import pandas as pd
import numpy as np
import ast
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD

skin = pd.read_csv('../data/skindata.csv', index_col=[0])

def content_recommender(product):
    skin_ing = skin[['Product', 'Product_id', 'Ingredients', 'Product_Url', 'Ing_Tfidf', 'Rating']]
    skin_ing.drop_duplicates(inplace=True)
    skin_ing = skin_ing.reset_index(drop=True)
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(skin_ing['Ingredients'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    titles = skin_ing[['Product', 'Ing_Tfidf', 'Rating']]
    indices = pd.Series(skin_ing.index, index=skin_ing['Product'])
    idx = indices[product]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    product_indices = [i[0] for i in sim_scores]
    return titles.iloc[product_indices]

content_recommender('Gold Camellia Beauty Oil')
