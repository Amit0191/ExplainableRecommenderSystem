from surprise import Dataset, accuracy
from surprise import KNNBasic
from surprise.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from scipy import spatial
from surprise import Trainset

data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.30)
similarityMatrix = np.zeros((10, 10))
# yourFile = 'venv\Lib\site-packages\surprise\prediction_algorithms\matrix.txt'
# np.savetxt('venv\Lib\site-packages\surprise\prediction_algorithms\matrix.txt', np.matrix(similarityMatrix))

#if os.path.isfile(yourFile) and os.access(yourFile, os.R_OK):
if True:
    print("Testset")
    print(type(testset[0]))
    algo = KNNBasic()
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions=predictions)
    accuracy.mae(predictions=predictions)
    accuracy.mse(predictions=predictions)
    tupleList = []

#for train in trainset.all_ratings():
#    print(train)

for prediction in predictions:
    # print(prediction)
    tupleList.append((int(prediction[0]), int(prediction[1]), int(prediction[3])))
sortedTupleL = sorted(tupleList)
midList = []
for predTup in sortedTupleL:
    if predTup[0] == 1 and predTup[2] >= 4:
        midList.append(predTup[1])
# print(midList)
movies = pd.read_csv('movies.dat', delimiter='::', engine='python', header=None)
movies.columns = ['MovieID', 'MovieName', 'Genre']
del movies['Genre']

# print(movies[movies['MovieID'] .isin(midList)])


