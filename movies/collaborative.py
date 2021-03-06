import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.model_selection import train_test_split

ratings = np.array(pd.read_csv("../data/ratings.csv", sep='\t', usecols=['user_id', 'movie_id', 'rating']))
movies = np.array(pd.read_csv("../data/movies.csv", sep='\t', usecols=['movie_id', 'title', 'genres']))

u_count = max(ratings[:, 0]) + 1 # number of users (plus 1 because array index start from 0)
m_count = max(ratings[:, 1]) + 1 # number of movies (plus 1 because array index start from 0)

ratings_train, ratings_test = train_test_split(ratings, test_size=0.05, random_state=42)
# Create user-item rating matrix 
r_mat = np.zeros([u_count,m_count])

for i in range(ratings_train.shape[0]):
    r_mat[int(ratings_train[i][0]), int(ratings_train[i][1])] = ratings_train[i][2]

# Mean of each user's rating
r_mean = np.zeros([u_count])

for i in range(u_count):
    r_mean[i] = np.mean(r_mat[i])

for i in range(u_count):
    for j in range(len(r_mat[i])):
        if r_mat[i][j] != 0:  
            r_mat[i][j] -= r_mean[i]

def predict_score(u, mov, k=5):
    # Find users who have rated 'mov'
    rated_u = np.array(np.nonzero(r_mat[:,mov]))
        
    if rated_u.shape[1] == 0 or (rated_u.shape[1] == 1 and rated_u[0,0] == u):
        #print("No users who have rated this movie")
        return 0
        
    rated_u = rated_u.flatten()
        
    sims = {}
    min_val = 0
    for i in range(rated_u.shape[0]):
        if rated_u[i] == u:
            continue
            
        sim = distance.cosine(r_mat[u,:],r_mat[rated_u[i],:])

        if len(sims) < k:
            sims[rated_u[i]] = sim


        # If the similarity of a current user is smaller than the minimum similarity of 'Sims',
        # then replace minimum user with current user.
        elif sim > sims[min_val]:
            sims.pop(min_val)
            sims[rated_u[i]] = sim

        min_val = min(sims, key=lambda k:sims[k])

    numerator = 0
    denominator = 0
    for item in sims:
        numerator += r_mat[item, mov] * sims[item]
        denominator += abs(sims[item])
    
    if denominator == 0:
        return 0
    return numerator / denominator

def evaluate_RMSE(test_set):
    n_tests = test_set.shape[0]
    SE = 0 # square error
    i = 0 
    for item in test_set:
        print(i)
        SE += (item[2] - predict_score(item[0], item[1]))**2
        i += 1
    
    RMSE = np.sqrt(SE/n_tests)
    print '\nCollaborative RMSE =', RMSE
    return RMSE
    

user_id = 1
predict_ratings = {}
user_ratings = r_mat[user_id, :].flatten().astype(float)

for i in range(1, len(user_ratings)):
    if user_ratings[i] == 0:
        predict_ratings[i] = predict_score(user_id, i)

recommended_movies_id = sorted(predict_ratings, key=lambda x: predict_ratings[x], reverse=True)
recommended_movies_id = recommended_movies_id[:12]

with open('../recommended/user/' + str(user_id) + '.txt', 'w+') as f:
    for item in recommended_movies_id:
        f.write("%s\n" % item)

# print evaluate_RMSE(ratings_test)