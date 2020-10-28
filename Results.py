from math import log2

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

directions = pd.read_csv("directions.dat", delimiter='::', engine='python')
directions.columns = ['MovieID', 'DirID']
directors = pd.read_csv("directors.dat", delimiter='::', engine='python')
directors.columns = ['DirID', 'Name', 'Popularity', 'DirGender', 'Birthday', 'Place']
del directors['Name']
del directors['Popularity']
del directors['Birthday']
del directors['Place']
ratings = pd.read_csv("ratings.dat", delimiter='::', engine='python', header=None)


def numDir(df
           , gender):
        if 'DirGender' in df.columns:
            return len(df[df['DirGender'] == gender])
        else:
            print("Provide data frame with DirGender column")


def topKRecomMtoF(predDF, n=0):
    numMaleList = []
    numFemaleList = []

    for i in range(1, 6041):
        uMovDF = predDF.loc[predDF['UserID'] == i]
        uMovDF.sort_values(by=['EstRatings'], ascending=False, inplace=True)
        uMovDF = uMovDF.head(n)
        # uMovID = uMovDF.loc[predDF['UserID'] == i].MovieID
        uMovID = uMovDF['MovieID']
        df = directions.loc[directions['MovieID'].isin(uMovID)]
        df = pd.merge(df, directors, on='DirID')
        numMale = numDir(df=df, gender=2)
        numFemale = numDir(df=df, gender=1)
        numMaleList.append(numMale)
        numFemaleList.append(numFemale)
    return numMaleList, numFemaleList, df


def createPredDf(filepath):
    predDF = pd.read_csv(filepath, delimiter=' ', header=None, engine='python')
    predDF.columns = ['UserID', 'MovieID', 'RealRatings', 'EstRatings']
    predDF.sort_values(by=['UserID'], inplace=True)
    return predDF


predDF = createPredDf("predictions.txt")
numMList, numFList, df = topKRecomMtoF(predDF=predDF, n=10)

# print(df)
predDF2 = createPredDf("predictions2.txt")
originalMList, originalFList, df = topKRecomMtoF(predDF=predDF2, n=10)

def counterForVisibility():
    nFusers = 0
    countF = 0
    nFUOrig = 0
    origCountF = 0
    sameCounter = 0
    for i in range(0, 6040):
        if numFList[i] > originalFList[i]:
            nFusers = nFusers + 1
            countF = countF + numFList[i] - originalFList[i]
        elif numFList[i] < originalFList[i]:
            nFUOrig = nFUOrig + 1
            origCountF = origCountF + originalFList[i] - numFList[i]
        else:
            sameCounter = sameCounter + 1
    return nFusers, countF, nFUOrig, origCountF, sameCounter


nFusers, countF, nFUOrig, origCountF, sameCounter = counterForVisibility()


def printVisibility(nFusers, countF, nFUOrig, origCountF, sameCounter):

    print(nFusers)
    print(countF)
    print(nFUOrig)
    print(origCountF)
    print(sameCounter)


# printVisibility(nFusers=nFusers, countF=countF, nFUOrig=nFUOrig, origCountF=origCountF,sameCounter=sameCounter)


def exposure(predDF, n):
    exposureListIndex = []
    exposureList = []

    for i in range(1, 6041):

        uMovDF = predDF.loc[predDF['UserID'] == i]
        uMovDF.sort_values(by=['EstRatings'], ascending=False, inplace=True)
        uMovDF = uMovDF.head(n)
        print(uMovDF)
        # uMovID = uMovDF.loc[predDF['UserID'] == i].MovieID
        uMovID = uMovDF['MovieID']
        # df = directions.loc[directions['MovieID'].isin(uMovID)]
        df = pd.merge(uMovDF, directions, on='MovieID')
        df = pd.merge(df, directors, on='DirID')
        dfGender = df[df['DirGender'] == 1]
        if len(dfGender.index) != 0:
            exposureListIndex.append(dfGender.index.tolist())
            exposureList.append(dfGender['MovieID'].values)
        # print(dfGender['MovieID'])
        # print(dfGender['MovieID'].index)
    return exposureListIndex, dfGender


x, dfGender = exposure(predDF=predDF, n=10)
y, dfGender2 = exposure(predDF=predDF2, n=10)


def exposureList(exposureListIndex):
    exposureList = []
    for i in exposureListIndex:
        for j in i:
            exposureVar = 1/log2(2 + j)
            exposureList.append(exposureVar)
    return exposureList


exposureListOur = exposureList(x)

exposureListOriginal = exposureList(y)

averageOur = sum(exposureListOur)/len(exposureListOur)
averageOriginal = sum(exposureListOriginal)/len(exposureListOriginal)
print(averageOur)
print(averageOriginal)
printVisibility(nFusers=nFusers, countF=countF, nFUOrig=nFUOrig, origCountF=origCountF, sameCounter=sameCounter)