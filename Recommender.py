import pandas as pd
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data = pd.read_csv("ratings.dat", delimiter='::', engine='python', header=None)
df_users = pd.read_csv('users.dat', delimiter='::', engine='python', header=None)
df_dir = pd.read_csv('directors.dat', delimiter='::', engine='python')
df_mov_dir = pd.read_csv('directions.dat', delimiter='::', engine='python')
df_movies = pd.read_csv('movies.dat', delimiter='::', engine='python', header=None)

data.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
df_users.columns = ['UserID', 'UserGender', 'Age', 'Occupation', 'Zip-code']
df_dir.columns = ['DirID', 'Name', 'Popularity', 'DirGender', 'Birthday', 'Place']
df_mov_dir.columns = ['MovieID', 'DirID']
df_movies.columns = ['MovieID', 'MovieName', 'Genre']

data = pd.merge(data, df_users, on='UserID')
data = pd.merge(data, df_mov_dir, on='MovieID')
data = pd.merge(data, df_dir, on='DirID')
movie_dir_gender = pd.merge(df_mov_dir, df_dir, on='DirID')

del data['Timestamp']
del data['Age']
del data['Occupation']
del data['Zip-code']
del data['Birthday']
del data['Place']
del data['Name']
del data['Popularity']

del movie_dir_gender['Birthday']
del movie_dir_gender['Place']
del movie_dir_gender['Name']
del movie_dir_gender['Popularity']

print(data.head())
# print(data[data.UserID == 1])
movie_dir_gender = movie_dir_gender.sort_values(by='MovieID')
movie_dir_gender = movie_dir_gender.reset_index(drop=True)
print(movie_dir_gender)
# no of male and female directors
fe_dir = len(movie_dir_gender[movie_dir_gender['DirGender'] == 1])
m_dir = len(movie_dir_gender[movie_dir_gender['DirGender'] == 2])
print(fe_dir)
print(m_dir)

mov_genre_gender = pd.merge(movie_dir_gender, df_movies, on='MovieID')
del mov_genre_gender['MovieName']

print(len(mov_genre_gender))
m_matrix = np.zeros((3696, 21), dtype=int)
c = 0
male_dir = 0
fem_dir = 0

for index, row in mov_genre_gender.iterrows():
    mID = row['MovieID']
    dID = row['DirID']
    dirGender = row['DirGender']
    genre = row['Genre']
    genre_row = genre.split('|')

    if index != (len(mov_genre_gender) - 1):
        nextMID = mov_genre_gender.iloc[index + 1][0]
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
                  17 if y == 'War' else 18 if y == 'Western' else y for y in genre_row]

    m_matrix[c][0] = mID
    for element in genre_list:
        m_matrix[c][element] = 1
        # m_matrix[][0] =
    if dirGender == 2:
        male_dir = male_dir + 1
        m_matrix[c][19] = male_dir
    elif dirGender == 1:
        fem_dir = fem_dir + 1
        m_matrix[c][20] = fem_dir
    if mID != nextMID:
        c = c + 1
        male_dir = 0
        fem_dir = 0


payoff_matrix = np.array(m_matrix)
average = np.sum(payoff_matrix[:, 1:], axis=1)
payoff_matrix = payoff_matrix[:, 1:] / average[:, None]

movieID = m_matrix[:, 0]
# movieID = [[i] for i in movieID]
# payoff_matrix = np.append(payoff_matrix, movieID, axis=1)
# Make a matrix for UserID 1
user_1 = data[data.UserID == 31]
print(user_1)
# print(user_1['MovieID'])
# print(len(user_1['MovieID']))
payoff_index = []
for x in user_1['MovieID']:
    movie_index = np.where(movieID == x)
    payoff_index.append(movie_index[0][0])

print(payoff_index)
# print(payoff_matrix[payoff_index])
M_for_user = payoff_matrix[payoff_index]
print(M_for_user)
print(len(M_for_user))
