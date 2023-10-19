import numpy as np
import pandas as pd
from scipy.spatial import distance
from abc import ABC, abstractmethod

#These classes receive a pandas dataframe, with userId, itemId and ratings, and try to predict
#the ratings for not present user-item interactions

class RatingEstimator(ABC): #abstrac class
    def __init__(self,ratings):
        self.original_ratings = ratings
        self.num_users = ratings['userId'].nunique()
        self.num_items = ratings['itemId'].nunique()
    #returns a pandas dataframe with only the predicted ratings for each user-item interaction
    #in a column called 'predicted_rating'
    @abstractmethod
    def get_predicted(self):
        pass


class MF(RatingEstimator):
    def __init__(self,ratings,factors=64,alpha=0.05,beta=0.01,iterations=100):
        super().__init__(ratings)
        self.original_ratings = super().ratings
        self.original_matrix = ratings.pivot(index='userId', columns='itemId', values='rating').to_numpy(na_value=0)
        self.num_users, self.num_items = self.original_matrix.shape
        self.factors = factors
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.factorized_matrix = self._factorize()
    
    def _factorize(self):
        #creating random matrix's to be used in the factorization
        self.user_matrix = np.random.normal(scale=1./self.factors,size=(self.num_users,self.factors))
        self.item_matrix = np.random.normal(scale=1./self.factors,size=(self.factors,self.num_items))
        for k in range(self.iterations):
            self._sgd()
        return np.dot(self.user_matrix,self.item_matrix)

    def _sgd(self):
        #stochastic gradient descent to aproach the error to 0
        for i in range(self.num_users):
            for j in range(self.num_items):
                #only computes based on the already given interactions
                if self.original_matrix[i,j]>0:
                    error = self.original_matrix[i,j] - self.user_matrix[i,:].dot(self.item_matrix[:,j])
                    user_temp = self.user_matrix[i,:].copy()
                    self.user_matrix[i,:] += self.alpha*(error*self.item_matrix[:, j] - self.beta*self.user_matrix[i, :])
                    self.item_matrix[:,j] += self.alpha*(error*user_temp - self.beta*self.item_matrix[:, j])

    def get_predicted(self):
        #"depivots" the factorized matrix so it turns back into a dataframe with all the ratings
        mf_ratings = pd.DataFrame(self.factorized_matrix)
        mf_ratings.index.name='userId'
        mf_ratings.columns.name='itemId'
        mf_ratings = pd.melt(mf_ratings.reset_index(),id_vars='userId',var_name='itemId',value_name='predicted_rating')
        #gets only the interactions not present in the original dataframe
        predicted_ratings = mf_ratings.merge(self.ratings, on=['userId', 'itemId'], how='left', indicator=True)
        return predicted_ratings[self.predicted_ratings['_merge'] == 'right_only'].drop(columns='_merge')


class UserKnn(RatingEstimator):
    def __init__(self,ratings,k=20):
        super().__init__(ratings)
        self.original_matrix = self.ratings.pivot(index='userId', columns='itemId', values='rating').to_numpy(na_value=0)
        self.knn = self.get_knn()
        self.k = k

    def get_knn(self):
        #centers the ratings around 0, so to minimize the impact of 0's in the euclidian distance
        center = lambda x: x - x.mean()
        centered_ratings = self.original_ratings.copy()
        centered_ratings['rating'] = center(self.original_ratings['rating'])
        centered_ratings = centered_ratings.pivot(index='userId', columns='itemId', values='rating').to_numpy(na_value=0)
        #this matrix will be the distance for each user to every other
        distances = np.zeros([self.num_users,self.num_users])
        #makes euclidian distance for each pair of users
        for i in range(self.num_users):
            for j in range(self.num_users):
                distances[i,j] = distance.euclidean(centered_ratings[i,:],centered_ratings[j,:])
                if i==j:
                    distances[i,j]=999999999
        #creates an array to store k nearest neighbors for every user
        knn = np.zeros([self.num_users,self.k])
        for i in range(self.num_users):
            #this gets the indexes for the k lowest values in a array
            knn[i,:] = np.argsort(distances[i,:])[:self.k]
        return knn
    
    def get_predicted(self):
        dictionary=[]
        for id in range(self.num_users):
            #user_ratings = predict_values(id,knn[id,:],user_item_matrix)
            user_ratings=[]
            #for each movie that hasn't been seen, calculates the rating according to the k nearest keighbors that did see that movie
            for j in range(self.num_items):
                if self.original_matrix[id,j] == 0:
                    user_ratings.append({'userId':id+1,'itemId':j+1,'predicted_rating':self.get_rating(id,j)})
            user_ratings = pd.DataFrame(user_ratings)
            #normalizes the value to be between 1 and 5
            min = user_ratings['predicted_rating'].min()
            max = user_ratings['predicted_rating'].max()
            user_ratings['predicted_rating'] = ((user_ratings['predicted_rating'] - min) / (max - min)) * 5
            #appends 
            user_ratings = user_ratings[['predicted_rating','itemId']]
            user_ratings['userId'] = id + 1
            dictionary.append(user_ratings)

        return pd.concat(dictionary, ignore_index=True)

    def get_rating(self,user,movie):
        sum=0
        count=0
        for neighbor in self.knn[user]:
            #it's quicker to access a value on a matrix then to search for it in a dataframe
            neighbor_rating = self.original_matrix[neighbor,movie]
            if neighbor_rating !=0:
                sum+=neighbor_rating
                count+=1
        return sum
        