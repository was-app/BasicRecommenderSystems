import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import estimators as estimators
pd.options.mode.chained_assignment = None #ommits an harmless warning

class Recommender:
    def __init__(self,ratings,method,num=10):
        self.method = method
        self.num_recommendations = num
        self.ratings = ratings
        self.predicted_ratings = self.predict_ratings()

    def predict_ratings(self):
        if(self.method=='mf'):
            estimator = estimators.MF(self.ratings)
            return estimator.get_predicted()
        if(self.method=='knn'):
            estimator = estimators.Knn(self.ratings)
            return estimator.get_predicted()
        if(self.method=='cb'):
            estimator = estimators.ContentBased(self.ratings)
            return estimator.get_predicted()

    def recommend(self):
        for user in self.ratings['userId'].unique():
        #gets the top n movies for the current user
            user_ratings = predicted[predicted['userId']==user]
            user_ratings.sort_values(by='predicted_rating',ascending=False,inplace=True)
            user_ratings = user_ratings.head(n=num_recommended)
            recommended = user_ratings['itemId'].to_numpy()
