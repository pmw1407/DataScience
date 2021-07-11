from os import sep
import numpy as np
import pandas as pd
import sys
from sklearn.decomposition import TruncatedSVD
import random

class Recommender:

    def __init__(self, test):
        self.test = test

    
    # Read Test Data
    def readTestData(self):
        self.testdf = pd.read_table(self.test, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'time_stamp'])

        self.testdf.drop('time_stamp', axis=1, inplace=True)
        self.testdf.drop('rating', axis=1, inplace=True)


    # Read Train Data
    def readTrainData(self, input):
        self.df = pd.read_table(input, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'time_stamp'])
        self.size = len(self.df)
        self.df.drop('time_stamp', axis=1, inplace =True)

        self.user_movie_rating = self.df.pivot_table('rating', index='user_id', columns='item_id')

        self.user_movie_rating.fillna(0, inplace=True)

        self.movie_user_rating = self.user_movie_rating.values.T

        SVD = TruncatedSVD(n_components=12)
        matrix = SVD.fit_transform(self.movie_user_rating)

        self.corr = np.corrcoef(matrix)


    # Predict
    def predict(self):
        ratepredict = []

        for tup in self.testdf.values:
            user_id = tup[0]
            item_id = tup[1]

            if item_id not in list(self.user_movie_rating.columns.values):
                ratepredict.append(random.randint(1,6))
                
            else:
                corr_idx = list(self.user_movie_rating.columns.values).index(item_id)
                rating = self.user_movie_rating.loc[user_id]

                div = 0
                rate_predict = 0
                for idx, rate in enumerate(rating.values):
                    
                    if rate != 0:
                        div += 1
                        corr = self.corr[corr_idx][idx]
                        if corr >= 0.5 :
                            rate_predict += rate
                        elif corr > 0 and corr < 0.5:
                            rate_predict += rate * corr
                        elif corr == 0:
                            rate_predict += random.randint(1, 5)
                        elif corr < 0:
                            rate_predict += 5 / rate
                
                ratepredict.append(rate_predict / div)

        ratepredict = np.array(ratepredict)
        self.testdf['rating'] = ratepredict


    # Write Output File
    def writeFile(self):
        fileNum = self.test[1]
        fileName = "u" + fileNum + ".base_prediction.txt"

        self.testdf.to_csv(fileName, header=None, index=False, sep='\t')
        

def main():
    train = sys.argv[1]
    test = sys.argv[2]
    recommender = Recommender(test)
    recommender.readTrainData(train)
    recommender.readTestData()
    recommender.predict()
    recommender.writeFile()


if __name__ == "__main__":
    main()