import pandas as pd
import numpy as np
import recommender as rm
import metrics as mt
from sklearn.model_selection import train_test_split
from pandas.api.extensions import no_default

ratings = pd.read_csv("datasets/ml-100k/u.data",sep='\t',names=["userId", "itemId", "rating", "timestamp"])
item_data = pd.read_csv("datasets/ml-100k/u.item",sep='|',encoding='latin-1',names=["itemId", "title", "release", "videoRelease","imdb","unknown","action","adventure","animation","childrens","comedy","crime","documentary","drama","fantasy","noir","horror","musical","mystery","romance","scifi","thriller","war","western"])
method = 'cb'
num=10

train, test = train_test_split(ratings)
recommender = rm.PredictionRec(train,item_data,method,num=num)

dictionary = []
for user in range(ratings['userId'].nunique()):
    recommendations = recommender.recommend(user+1)
    for j in range(num):
        dictionary.append ({'userId':user+1,'itemId':recommendations[j]})
recommended = pd.DataFrame(dictionary)

error_metrics = mt.ErrorMetrics(test,recommended,num)
print(error_metrics.precision,error_metrics.recall,error_metrics.f1)