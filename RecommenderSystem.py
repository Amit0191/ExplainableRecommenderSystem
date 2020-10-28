import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable
from surprise import Dataset, accuracy
from surprise import KNNBasic
from surprise.model_selection import train_test_split
from scipy import spatial
import nashpy as nash
import warnings
warnings.filterwarnings("ignore")
# check edit at line 306
from surprise.prediction_algorithms.knns import KNNBasic2
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class DataHandling:

    def __init__(self, fileName, header=''):
        if header is None:
            self.df = pd.read_csv(fileName, delimiter='::', engine='python', header=None)
        else:
            self.df = pd.read_csv(fileName, delimiter='::', engine='python')

    def setCols(self, colName):
        self.df.columns = colName

    def delCol(self, colName):
        del self.df[colName]

    def delCols(self, colList):
        for colName in colList:
            del self.df[colName]


class DfCalculations:
    def __init__(self):
        pass

    # Number of directors in a DataFrame.
    def numDir(self, df, gender):
        if 'DirGender' in df.columns:
            return len(df[df['DirGender'] == gender])
        else:
            print("Provide data frame with DirGender column")

    # Percentage of Male Directors.
    def percentMDir(self, numMDir, numFDir):
        return numMDir/(numMDir+numFDir)*100

    # Percentage of Female Directors.
    def percentFDir(self, numMDir, numFDir):
        return numFDir/(numMDir+numFDir)*100

    # Merge DataFrames
    def mergeDf(self, df1, df2, onID):
        return pd.merge(df1, df2, on=onID)

    # Sort a DataFrame
    def sortValues(self, df, colName):
        df = df.sort_values(by=colName)
        return df.reset_index(drop=True)

    # Returns normalized item X feature matrix
    def payoffMatrix(self, df):
        mMatrix = np.zeros((4000, 21), dtype=int)
        c = 0
        mDir = 0
        fDir = 0
        for index, row in df.iterrows():
            mID = row['MovieID']
            dID = row['DirID']
            dirGender = row['DirGender']
            genre = row['Genre']
            genreRow = genre.split('|')

            if index != (len(df) - 1):
                nextMID = df.iloc[index + 1][0]
            else:
                nextMID = -1

            genre_list = [1 if y == 'Action' else 2 if y == 'Adventure' else
                          3 if y == 'Animation' else 4 if y == "Children's" else
                          5 if y == 'Comedy' else 6 if y == 'Crime' else
                          7 if y == 'Documentary' else 8 if y == 'Drama' else
                          9 if y == 'Fantasy' else 10 if y == 'Film-Noir' else
                          11 if y == 'Horror' else 12 if y == 'Musical' else
                          13 if y == 'Mystery' else 14 if y == 'Romance' else
                          15 if y == 'Sci-Fi' else 16 if y == 'Thriller' else
                          17 if y == 'War' else 18 if y == 'Western' else y for y in genreRow]
            mMatrix[c][0] = mID
            for element in genre_list:
                mMatrix[c][element] = 1
                # m_matrix[][0] =
            if dirGender == 2:
                mDir = mDir + 1
                mMatrix[c][19] = mDir
            elif dirGender == 1:
                fDir = fDir + 1
                mMatrix[c][20] = fDir
            if mID != nextMID:
                c = c + 1
                mDir = 0
                fDir = 0

        payoffMatrix = np.array(mMatrix)
        average = np.sum(payoffMatrix[:, 1:], axis=1)
        payoffMatrix = payoffMatrix[:, 1:] / average[:, None]
        return payoffMatrix, mMatrix

    # Payoff Matrix for a specific User.
    def userPayoff(self, payoffMatrix, userMList, movieIDList):
        payoffIndex = []
        for movie in userMList['MovieID']:
            movieIndex = np.where(movieIDList == movie)
            payoffIndex.append(movieIndex[0][0])
        return payoffMatrix[payoffIndex], payoffIndex

    # Weighted Payoff Matrix taking rating of the user for a specific item.
    def userWeightedPayoff(self, payoffMatrix, userMList, movieIDList, multiplier):

        payoffIndex = []
        for movie in userMList['MovieID']:
            movieIndex = np.where(movieIDList == movie)
            payoffIndex.append(movieIndex[0][0])
        c = 0
        weightedPayoff = payoffMatrix[payoffIndex]
        for movie, rating in zip(userMList['MovieID'], userMList['Rating']):
            if c == 19:
                weightedPayoff[c] = weightedPayoff[c] * rating
            elif c == 18:
                weightedPayoff[c] = weightedPayoff[c] * rating
            else:
                weightedPayoff[c] = weightedPayoff[c] * rating
            c = c + 1

        return weightedPayoff, payoffIndex

    # Returns a list of Movie IDs.
    def midList(self, mMatrix):
        return mMatrix[:, 0]

    # Return a list of Movie IDs seen by a user U.
    def userMidList(self, df, userID):
        return df[df.UserID == userID]

    # Normalie rating by subtracting mean from the ratings.
    def adjustedRanking(self, df):
        mean = df['Rating'].mean()
        df['AdjustedRating'] = df['Rating'] - mean
        return df


class SurpriseRecommender:
    def __init__(self):
        self.data = Dataset.load_builtin('ml-1m')

    def trainTestSplit(self, test_size):
        return train_test_split(self.data, test_size=test_size)

    # Returns a trainsetDF containing columns UserID, MovieID, Rating
    def trainsetDf(self, trainset):
        uidList = []
        iidList = []
        rList = []
        # trainset.all_ratings returns inner user and item ids.
        for allRows in trainset.all_ratings():
            (uid, iid, rating) = allRows
            uidList.append(int(trainset.to_raw_uid(uid)))
            iidList.append(int(trainset.to_raw_iid(iid)))
            rList.append(rating)
        trainsetDf = pd.DataFrame(list(zip(uidList, iidList, rList)),
                                  columns=['UserID', 'MovieID', 'Rating'])

        return trainsetDf

    # Accepts a train and test dataset and returns a prediction Object.
    def prediction(self, trainset, testset):
        algo = KNNBasic()
        algo.fit(trainset)
        predictions = algo.test(testset)
        return predictions

    # If estimated rating is off by a threshold remove it from the predictions.
    def refinePrediction(self, predictions):
        for prediction in predictions:
            if abs(prediction[3]) > 4.0:
                predictions.remove(prediction)
        return predictions

    def originalKNNPred(self, trainset, testset):
        algo = KNNBasic2()
        algo.fit(trainset)
        predictions2 = algo.test(testset)
        return predictions2


class LinearProgramming:
    def __init__(self, x20=0):

        self.model = LpProblem(name="GameTheoryRecommender", sense=LpMaximize)
        self.x1 = LpVariable(name="x1", lowBound=0)
        self.x2 = LpVariable(name="x2", lowBound=0)
        self.x3 = LpVariable(name="x3", lowBound=0)
        self.x4 = LpVariable(name="x4", lowBound=0)
        self.x5 = LpVariable(name="x5", lowBound=0)
        self.x6 = LpVariable(name="x6", lowBound=0)
        self.x7 = LpVariable(name="x7", lowBound=0)
        self.x8 = LpVariable(name="x8", lowBound=0)
        self.x9 = LpVariable(name="x9", lowBound=0)
        self.x10 = LpVariable(name="x10", lowBound=0)
        self.x11 = LpVariable(name="x11", lowBound=0)
        self.x12 = LpVariable(name="x12", lowBound=0)
        self.x13 = LpVariable(name="x13", lowBound=0)
        self.x14 = LpVariable(name="x14", lowBound=0)
        self.x15 = LpVariable(name="x15", lowBound=0)
        self.x16 = LpVariable(name="x16", lowBound=0)
        self.x17 = LpVariable(name="x17", lowBound=0)
        self.x18 = LpVariable(name="x18", lowBound=0)
        self.x19 = LpVariable(name="x19", lowBound=0, upBound=0.3)
        self.x20 = LpVariable(name="x20", lowBound=x20)
        self.w = LpVariable(name="w")
        self.varList = np.array([self.x1, self.x2, self.x3, self.x4, self.x5, self.x6, self.x7, self.x8,
                                self.x9, self.x10, self.x11, self.x12, self.x13, self.x14, self.x15,
                                self.x16, self. x17, self.x18, self.x19, self.x20])

    # Construct a Linear Programming model with relevant constraints.
    def constructModel(self, payoffMForUser):
        self.model += (self.x1 + self.x2 + self.x3 + self.x4 + self.x5 + self.x6 + self.x7 + self.x8 + self.x9 +
                      self.x10 + self.x11 + self.x12 + self.x13 + self.x14 + self.x15 + self.x16 + self.x17 +
                      self.x18 + self.x19 + self.x20 == 1)
        self.model += self.w
        for i in range(len(payoffMForUser)):
            row = np.array(payoffMForUser[i, :])
            constraint_v = self.varList.dot(row)
            self.model += (self.w - constraint_v <= 0)
        return self.model

    # Solve the model and return a list of probability Distribution.
    def solveModel(self, model, payoffMForUser):
        probDistList = [0] * 20
        model.solve()
        for var in model.variables():
            # print(f"{var.name}: {var.value()}")
            if var.name[0] == 'x':
                index = int(var.name[1:]) - 1
                probDistList[index] = var.value()
        return probDistList, self.model.variables()[0].value()

    # def constructAndSolveModel(self, payoffMForUser, name):
    #     self.model += (self.x1 + self.x2 + self.x3 + self.x4 + self.x5 + self.x6 + self.x7 + self.x8 + self.x9 +
    #                    self.x10 + self.x11 + self.x12 + self.x13 + self.x14 + self.x15 + self.x16 + self.x17 +
    #                    self.x18 + self.x19 + self.x20 == 1, name)
    #     self.model += self.w
    #     probDistList = [0] * 20
    #     for i in range(len(payoffMForUser)):
    #         row = np.array(payoffMForUser[i, :])
    #         constraint_v = self.varList.dot(row)
    #         self.model += (self.w - constraint_v <= 0)
    #     # Model constructed
    #     x = True
    #     while x is True:
    #         self.model.solve()
    #         if int(self.model.variables()[0]) == 0:
    #             pass

    def userProfile(self):
        pass

    def itemProfile(self):
        pass


def datahandling():
    directions = DataHandling(fileName="directions.dat")
    directions.setCols(['MovieID', 'DirID'])

    directors = DataHandling(fileName="directors.dat")
    directors.setCols(['DirID', 'Name', 'Popularity', 'DirGender', 'Birthday', 'Place'])
    directors.delCols(['Name', 'Popularity', 'Birthday', 'Place'])

    movies = DataHandling(fileName="movies.dat", header=None)
    movies.setCols(['MovieID', 'MovieName', 'Genre'])
    movies.delCol('MovieName')

    # ratings = DataHandling(fileName="ratings.dat", header=None)
    ratings = DataHandling(fileName="u.data", header=None)
    ratings.setCols(['UserID', 'MovieID', 'Rating', 'Timestamp'])
    ratings.delCol('Timestamp')

    users = DataHandling(fileName="users.dat", header=None)
    users.setCols(['UserID', 'UserGender', 'Age', 'Occupation', 'Zip-code'])
    users.delCols(['Age', 'Occupation', 'Zip-code'])

    return directions.df, directors.df, movies.df, ratings.df, users.df


directions, directors, movies, ratings, users = datahandling()
surprise = SurpriseRecommender()
trainset, testset = surprise.trainTestSplit(test_size=0.3)
trainsetDf = surprise.trainsetDf(trainset=trainset)


# Function returns all merge Dfs, allDf has rows corresponding to trainset given by surprise
def dfCalculator(trainsetDf):
    movDirGen = DfCalculations().mergeDf(directions, directors, onID='DirID')
    # Pass this to payoff matrix
    movGenDirGen = DfCalculations().mergeDf(df1=movDirGen, df2=movies, onID='MovieID')
    allDf = DfCalculations().mergeDf(df1=trainsetDf, df2=users, onID='UserID')
    allDf = DfCalculations().mergeDf(df1=allDf, df2=directions, onID='MovieID')
    allDf = DfCalculations().mergeDf(df1=allDf, df2=directors, onID='DirID')

    trainsetDf = DfCalculations().sortValues(trainsetDf, colName='UserID')
    return movDirGen, movGenDirGen, allDf, trainsetDf


movDirGen, movGenDirGen, allDf, trainsetDf = dfCalculator(trainsetDf)


def printDFs():
    print(directions.head())
    print("----------------------")
    print(directors.head())
    print("---------------------------------------------")
    print(movies.head())
    print("---------------------------------------------")
    print(ratings.head())
    print("--------------------------------")
    print(users.head())
    print("------------------------------")
    print(movDirGen.head())
    print("----------------------------------------")
    print(movGenDirGen.head())
    print("-------------------------------------------------------")
    print(allDf.head())
    print("-------------------------------------------------------")
    print(trainsetDf.head())
    # allDf = mdg.mergeDf(df1=ratings.df, df2=users.df, onID='UserID')
    print("--------------------------------------------------------------------")
    print(allDf.head())
    print("All DataFrames done")
    print('---------------------------------------------------------------------')
    print("Percentage of female directors in the database : ")
    print("{:.2f}".format(DfCalculations().numDir(directors, 1) / (DfCalculations().numDir(directors, 2)
                                                                   + DfCalculations().numDir(directors, 1)) * 100))
    print("Percentage of movies directed by female directors :")
    print("{:.2f}".format(DfCalculations().numDir(movDirGen, 1) / (DfCalculations().numDir(movDirGen, 2)
                                                                   + DfCalculations().numDir(movDirGen, 1)) * 100))
    print("-------------------------------------------------------")


printDFs()


# Calculates weightedPayoff and returns a weighted PayoffList for every User.
def weightedPayoff():

    dfCalculations = DfCalculations()
    payoffMatrix, mMatrix = dfCalculations.payoffMatrix(movGenDirGen)

    movIdList = dfCalculations.midList(mMatrix)
    weightedPayoffList = []

    for i in range(1, 6041):
        userMList = dfCalculations.userMidList(allDf, i)
        userMList = dfCalculations.adjustedRanking(userMList)
        userPayoff, userPayoffIndex = dfCalculations.userPayoff(payoffMatrix=payoffMatrix, userMList=userMList,
                                                                movieIDList=movIdList)
        weightedPayoff, weightedIndex \
            = dfCalculations.userWeightedPayoff \
              (payoffMatrix=payoffMatrix, userMList=userMList, movieIDList=movIdList, multiplier=20.0)
        weightedPayoffList.append(weightedPayoff)
    return weightedPayoffList


weightedPayoffList = weightedPayoff()


# predictions = surprise.refinePrediction(surprise.trainTest(trainset=trainset, testset=testset), 1)
# predictions = surprise.trainTest(trainset=trainset, testset=testset)
# model = linearProgramming.construct_model(payoffMForUser=userPayoff)
# linearProgramming.solveModel(model)
def solveLinearProgram():
    probDistMatrix = []
    probDistList = []
    ij = 0
    for userP in weightedPayoffList:
        ij = ij + 1
        if ij % 10 == 0:
            print("--------------------------------------------------------------------------")
            print(ij)
            print("--------------------------------------------------------------------------")
        x = True
        while x is True:

            linearProgramming = LinearProgramming()
            model = linearProgramming.constructModel(payoffMForUser=userP)
            probDistList, w = linearProgramming.solveModel(model=model, payoffMForUser=userP)
            if int(w) == 0 and (len(userP) != 0):
                userP = np.delete(userP, 0, 0)
            else:
                x = False
        probDistMatrix.append(probDistList)
    return probDistMatrix


probDistMatrix = solveLinearProgram()
print(probDistMatrix)

print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')


def solveLinearProgram2():

    probDistMatrix2 = []
    probDistList2 = []
    ij = 0
    for userP in weightedPayoffList:
        ij = ij + 1
        if ij % 10 == 0:
            print("--------------------------------------------------------------------------")
            print("--------------------------------------------------------------------------")
        x = True
        while x is True:
            linearProgramming2 = LinearProgramming(x20=0.20)
            model = linearProgramming2.constructModel(payoffMForUser=userP)
            probDistList2, w2 = linearProgramming2.solveModel(model=model, payoffMForUser=userP)
            if int(w2) == 0 and (len(userP) != 0):
                userP = np.delete(userP, 0, 0)
            else:
                x = False
        probDistMatrix2.append(w2)
    return probDistMatrix2


probDistMatrix2 = solveLinearProgram2()


def computeSimMatrix():
    similarityMatrix = np.zeros((6040, 6040))
    # Compute Similarity Matrix
    for i in range(0, 6040):
        if i % 500 == 0:
            print(i)
        for j in range(0, 6040):
            similarity = 1 - spatial.distance.cosine(probDistMatrix[i], probDistMatrix[j])
            similarity = round(similarity, 5)
            similarityMatrix[i][j] = similarity
    # user_5 = (mdg.userMList(allDf, 5))
    # print(user_5[user_5['Rating'] >= 4])
    return similarityMatrix


def computeBetterSimilarity():

    similarityMatrix = np.zeros((6040, 6040))
    # Compute Similarity Matrix
    for i in range(0, 6040):
        if i % 500 == 0:
            print(i)

        for j in range(0, 6040):
            similarity = 1 - spatial.distance.cosine(probDistMatrix2[i], probDistMatrix[j])
            similarityMatrix[i][j] = similarity
    return similarityMatrix


# This line has been changed.
similarityMatrix = computeBetterSimilarity()
np.savetxt('matrix.txt', np.matrix(similarityMatrix))

predictions = surprise.prediction(trainset=trainset, testset=testset)
predictions2 = surprise.originalKNNPred(trainset=trainset, testset=testset)
# predictions = surprise.refinePrediction(predictions=predictions)


#
# def printRest():
#     for row in range(0, 5):
#         print(weightedPayoffList[row])
#     print("----------------------------------------------------------------------")
#     for row in range(0, 5):
#         print(similarityMatrix[row][:])
#     print("----------------------------------------------------------------------")
#     #for predict in predictions:
#     #    print(predict)
#
#
# printRest()


print(accuracy.rmse(predictions=predictions))
print(accuracy.mae(predictions=predictions))
print(accuracy.mse(predictions=predictions))
print(accuracy.rmse(predictions=predictions2))
print(accuracy.mae(predictions=predictions2))
print(accuracy.mse(predictions=predictions2))

predTupleList = []
predMovList = []

for prediction in predictions:
    # print(prediction)
    predTupleList.append((int(prediction[0]), int(prediction[1]), int(prediction[3])))
    predMovList.append(int(prediction[1]))

with open('predictions.txt', 'w') as f:
    for item in predictions:
        f.write("%s %s %s %s\n" % (str(item[0]), str(item[1]), str(item[2]), str(item[3])))

with open('predictions2.txt', 'w') as f:
    for item2 in predictions2:
        f.write("%s %s %s %s\n" % (str(item2[0]), str(item2[1]), str(item2[2]), str(item2[3])))




