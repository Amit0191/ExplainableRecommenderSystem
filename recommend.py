from surprise import SVD
from surprise import Dataset
from surprise import KNNBasic
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Load the movielens-100k dataset (download it if needed),
df_mov_dir = pd.read_csv('directions.dat', delimiter='::', engine='python')
df_mov_dir.columns = ['MovieID', 'DirID']
df_dir = pd.read_csv('directors.dat', delimiter='::', engine='python')
df_dir.columns = ['DirID', 'Name', 'Popularity', 'DirGender', 'Birthday', 'Place']

movie_dir_gender = pd.merge(df_mov_dir, df_dir, on='DirID')

del movie_dir_gender['Birthday']
del movie_dir_gender['Place']
del movie_dir_gender['Name']
del movie_dir_gender['Popularity']

data = Dataset.load_builtin('ml-100k')

trainset, testset = train_test_split(data, test_size=.30)
# We'll use the famous SVD algorithm.
algo = KNNBasic()
# Run 5-fold cross-validation and print results
# print(cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True))
algo.fit(trainset)
predictions = algo.test(testset)
# print(predictions[0])
c = 0
mList = []
userMList = []
for prediction in predictions:
    if abs(prediction[2] - prediction[3]) > 1:
        predictions.remove(prediction)
        c = c + 1
    else:
        mList.append(prediction[1])
        print(prediction)
# for prediction in predictions:
#    if prediction[0] == "5" and (abs(prediction[2] - prediction[3]) < 1):
#        print(prediction)
#        userMList.append(prediction[1])

user_recommended_movies = movie_dir_gender[movie_dir_gender['MovieID'].isin(userMList)]
# print(userMList)
user_f_dir = len(user_recommended_movies[user_recommended_movies['DirGender'] == 1])
user_m_dir = len(user_recommended_movies[user_recommended_movies['DirGender'] == 2])
# print(user_f_dir)
# print(user_m_dir)
# print(user_f_dir/(user_m_dir + user_f_dir)*100)
# print(user_m_dir/(user_m_dir + user_f_dir)*100)
# mSeries = movie_dir_gender[movie_dir_gender.MovieID == movie_id]

all_recommended_movies = movie_dir_gender[movie_dir_gender['MovieID'].isin(mList)]
# print(all_recommended_movies)
# fe_dir = len(all_recommended_movies[all_recommended_movies['DirGender'] == 1])
# m_dir = len(all_recommended_movies[all_recommended_movies['DirGender'] == 2])

# print(fe_dir)
# print(m_dir)
# print(fe_dir/(m_dir+fe_dir)*100)

fe_dir = 0
m_dir = 0
# c = 0
# for recMov in mList:
#    mListpd = movie_dir_gender[movie_dir_gender['MovieID'] == int(recMov)]
#    fe_dir = fe_dir + len(mListpd[mListpd['DirGender'] == 1])
#    m_dir = m_dir + len(mListpd[mListpd['DirGender'] == 2])
#
# print(fe_dir)
# print(m_dir)
# print(fe_dir/(m_dir+fe_dir)*100)
# print(m_dir/(m_dir+fe_dir)*100)
for prediction in predictions:
    if abs(prediction[2] - prediction[3]) < 1:
        userMList.append(prediction[1])
        mListpd = movie_dir_gender[movie_dir_gender['MovieID'] == int(prediction[1])]
        fe_dir = fe_dir + len(mListpd[mListpd['DirGender'] == 1])
        m_dir = m_dir + len(mListpd[mListpd['DirGender'] == 2])
print(fe_dir)
print(m_dir)
print(fe_dir/(m_dir+fe_dir)*100)
print(m_dir/(m_dir+fe_dir)*100)
print(len(predictions))
