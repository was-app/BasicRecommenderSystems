import numpy as np
import pandas as pd
import estimators as estimators
from scipy.spatial import distance
from abc import ABC, abstractmethod

class Recommender(ABC):
    def __init__(self,ratings,num):
        self.ratings = ratings
        self.num_recommendations = num

    @abstractmethod
    def recommend(self):
        pass


#Recommends based on similar content from a movie
class SimilarityRec(Recommender):
    def __init__(self,ratings,num=10):
        super().__init__(ratings,num)
        self.original_matrix = ratings.pivot(index='userId', columns='itemId', values='rating').to_numpy(na_value=0)
        self.num_items = ratings['itemId'].nunique()
        self.set_similarity()
    
    def set_similarity(self):
        self.similarity_matrix = np.zeros(self.num_items,self.num_items)
        for i in range(self.num_items):
            for j in range(self.num_items):
                self.similarity_matrix[i,j] = distance.cosine(self.original_matrix[:,i],self.original_matrix[:,j])
                if(i==j):
                    self.similarity_matrix[i,j] = 9999999
    def recommend(self,movieId):
        similar_movies = self.similarity_matrix[movieId-1,:]
        similar_movies = np.sort(similar_movies)
        return similar_movies[1:self.num_recommendations]


#Recomends based on prediction of ratings for non present interactions
class PredictionRec(Recommender):
    def __init__(self,ratings,method,num=10):
        super().__init__(ratings,num)
        self.method = method
        self.predicted_ratings = self.predict_ratings()

    def predict_ratings(self):
        if(self.method=='mf'):
            estimator = estimators.MF(self.ratings)
            return estimator.get_predicted()
        if(self.method=='knn'):
            estimator = estimators.UserKnn(self.ratings)
            return estimator.get_predicted()
        if(self.method=='cb'):
            estimator = estimators.ContentBased(self.ratings)
            return estimator.get_predicted()

    #return an array of n recommended movie's ids for each user
    def recommend(self,userId):
        #gets the top n movies for the user
        user_ratings = self.predicted_ratings[self.predicted_ratings['userId']==userId]
        user_ratings.sort_values(by='predicted_rating',ascending=False,inplace=True)
        user_ratings = user_ratings.head(n=self.num_recommendations)
        return user_ratings['itemId'].to_numpy()