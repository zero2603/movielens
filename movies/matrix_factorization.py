import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.model_selection import train_test_split

class MatrixFactorization(object):
    def __init__(self, Y_data, K, lambda_param = 0.1, X_init = None, W_init = None,
        learning_rate = 0.5, loop = 1000):
        self.Y_raw_data = Y_data
        self.K = K

        self.lambda_param = lambda_param
        # learning rate in gradient descent
        self.learning_rate = learning_rate
        # numbers of loop 
        self.loop = loop

        self.n_users = int(np.max(Y_data[:, 0])) + 1
        self.n_items = int(np.max(Y_data[:, 1])) + 1
        self.n_ratings = Y_data.shape[0]

        if X_init is None:
            self.X = np.random.randn(self.n_items, K)
        else:
            self.X = X_init
        
        if W_init is None:
            self.W = np.random.randn(K, self.n_users)
        else:
            self.W = W_init

        # normalized data, update later in normalized_Y function
        self.Y_data_normalized = self.Y_raw_data.copy()

    def normalize_Y(self):
        user_col = 0
        item_col = 1
        n_objects = self.n_users
        
        users = self.Y_raw_data[:, user_col]
        self.mu = np.zeros((n_objects,)) 
        for n in range(n_objects):
            ids = np.where(users == n)[0].astype(np.int32)
            item_ids = self.Y_data_normalized[ids, item_col]
            ratings = self.Y_data_normalized[ids, 2]

            m = np.mean(ratings) 
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[n] = m
            # normalize
            self.Y_data_normalized[ids, 2] = ratings - self.mu[n]
    
    def loss(self):
        L = 0 
        for i in range(self.n_ratings):
            # user, item, rating
            n, m, rate = int(self.Y_data_normalized[i, 0]), int(self.Y_data_normalized[i, 1]), self.Y_data_normalized[i, 2]
            L += 0.5*(rate - self.X[m, :].dot(self.W[:, n]))**2
        
        # take average
        L /= self.n_ratings
        # regularization, don't ever forget this 
        L += 0.5 * self.lambda_param * (np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro'))
        return L 
    
    def get_items_rated_by_user(self, user_id):
        """
        get all items which are rated by user user_id, and the corresponding ratings
        """
        ids = np.where(self.Y_data_normalized[:,0] == user_id)[0] 
        item_ids = self.Y_data_normalized[ids, 1].astype(np.int32) # indices need to be integers
        ratings = self.Y_data_normalized[ids, 2]
        return (item_ids, ratings)
        
        
    def get_users_who_rate_item(self, item_id):
        """
        get all users who rated item item_id and get the corresponding ratings
        """
        ids = np.where(self.Y_data_normalized[:,1] == item_id)[0] 
        user_ids = self.Y_data_normalized[ids, 0].astype(np.int32)
        ratings = self.Y_data_normalized[ids, 2]
        return (user_ids, ratings)

    def updateX(self):
        for m in range(self.n_items):
            user_ids, ratings = self.get_users_who_rate_item(m)
            W_m = self.W[:, user_ids]
            # gradient
            grad_xm = -(ratings - self.X[m, :].dot(W_m)).dot(W_m.T)/self.n_ratings + self.lambda_param * self.X[m, :]
            self.X[m, :] -= self.learning_rate *  grad_xm.reshape((self.K,))
    
    def updateW(self):
        for n in range(self.n_users):
            item_ids, ratings = self.get_items_rated_by_user(n)
            X_n = self.X[item_ids, :]
            # gradient
            grad_wn = -X_n.T.dot(ratings - X_n.dot(self.W[:, n]))/self.n_ratings + self.lambda_param * self.W[:, n]
            self.W[:, n] -= self.learning_rate * grad_wn.reshape((self.K,))
    
    def fit(self):
        self.normalize_Y()
        for i in range(self.loop):
            self.updateX()
            self.updateW()

    def predict(self, u, i):
        u = int(u)
        i = int(i)
        bias = self.mu[u]

        pred = self.X[i, :].dot(self.W[:, u]) + bias 
        # truncate if results are out of range [0, 5]
        if pred < 0:
            return 0 
        if pred > 5: 
            return 5 
        return pred 

    def predict_for_user(self, user_id):
        ids = np.where(self.Y_data_normalized[:, 0] == user_id)[0]
        items_rated_by_u = self.Y_data_normalized[ids, 1].tolist()

        y_pred = self.X.dot(self.W[:, user_id]) + self.mu[user_id]
        predicted_ratings = {}
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                predicted_ratings[i] = y_pred[i]
        
        return predicted_ratings

    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0 # squared error
        for n in range(n_tests):
            pred = self.predict(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2])**2 

        RMSE = np.sqrt(SE/n_tests)
        return RMSE

ratings = pd.read_csv("../data/ratings.csv", sep='\t', usecols=['user_id', 'movie_id', 'rating'])
ratings = ratings.as_matrix()

# indices in Python start from 0
ratings[:, :2] -= 1 

rate_train, rate_test = train_test_split(ratings, test_size=0.05, random_state=42)

rs = MatrixFactorization(rate_train, K = 2, lambda_param = 0.1, learning_rate = 0.5, loop = 20)
rs.fit()
# evaluate on test data
RMSE = rs.evaluate_RMSE(rate_test)
print '\nItem-based MF, RMSE =', RMSE
# user_id = 3
# predict_ratings = rs.predict_for_user(user_id)
# recommended_movie_ids = sorted(predict_ratings, key=lambda x: predict_ratings[x], reverse=True)
# recommended_movie_ids = recommended_movie_ids[:12]

# with open('../recommended/mf/' + str(user_id) + '.txt', 'w+') as f:
#     for item in recommended_movie_ids:
#         f.write("%s\n" % item)