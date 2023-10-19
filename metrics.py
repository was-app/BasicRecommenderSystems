import numpy as np


class ErrorMetrics:
    #ratings = original ratings
    def __init__(self,test,recommended,num):
        self.test = test
        self.recommended = recommended
        self.num_recommended = num
        self.precision,self.recall,self.f1 = self.measure_errors()


#based on number of hits and misses, not on ratings
def measure_errors(self): #predicted = a DF with predicted ratings for interactions not present in The Training DF
    precision = 0
    recall = 0
    f1 = 0
    for user in self.test['userId'].unique():
        curr_precision,curr_recall,hits = 0,0,0
        watched_movies = self.test[self.test['userId']==user]
        watched_movies = watched_movies['itemId'].to_numpy()
        anterior = hits
        for movie in self.recommended['userId'==user]:
            if np.isin(movie,watched_movies):
                hits+=1
        curr_precision = hits/self.num_recommended
        precision += curr_precision
        curr_recall = hits/len(watched_movies)
        recall += curr_recall
        if (curr_precision+curr_recall) == 0:
            f1+=0
        else:
            f1 += 2*(curr_precision*curr_recall)/(curr_precision+curr_recall)
    precision = precision/self.test['userId'].nunique()
    recall = recall/self.test['userId'].nunique()
    f1 = f1/self.test['userId'].nunique()
    return precision, recall, f1