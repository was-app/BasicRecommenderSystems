import pandas as pd
import numpy as np
import recommender as rm
import metrics
from sklearn.model_selection import train_test_split
from pandas.api.extensions import no_default

ratings = pd.read_csv("datasets/ml-100k/u.data",sep='\t',names=["userId", "itemId", "rating", "timestamp"])
item_data = pd.read_csv("datasets/ml-100k/u.item",sep='|',encoding='latin-1',names=["itemId", "title", "release", "videoRelease","imdb","unknown","action","adventure","animation","childrens","comedy","crime","documentary","drama","fantasy","noir","horror","musical","mystery","romance","scifi","thriller","war","western"])

num=10
train, test = train_test_split(ratings)
recommender = rm.PredictionRec(train,'mf',num=num)

dictionary = []
for user in ratings['userId'].unique():
    recommendations = recommender.recommend(user)
    for j in range(num):
        dictionary.append ({'userId':user,'itemId':recommendations[j]})
recommended = pd.concat(dictionary,ignore_index=True)



error_metrics = metrics.ErrorMetrics(test,recommended,num)