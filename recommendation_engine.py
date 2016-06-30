from math import*
import pandas as pd
import numpy as np

class RecommendationEngine:
    MAX_RECS = 3
    MAX_RATING = 5
    lambda_ = 0.1
    num_factors = 50
    n_iterations = 200

    def __init__(self):
        self.name = 'recommendation engine'

    def generate_recommendations(self, user_id):
        # load data
        # done in-line with each web request.  Far from ideal and this pattern would never reach prod
        users, movies, ratings = self.load_data()
        user_exists = False
        all_user_ids = users.id.values[:]

        # ensure user requested exists in database
        # if not, bubble this info to resource layer which will throw the appropriate http error
        if user_id in all_user_ids:
            user_exists = True

            # get list of users who we want to generate similarity scores for
            comp_user_ids =  np.array(filter(lambda x: x != user_id, all_user_ids))

            # create a dataframe that keeps track of the similarity measures between all other users and the requested user
            df_sim_scores = pd.DataFrame(columns=['id', 'jac', 'cos', 'comb'])

            user_movies_rated = ratings[ratings['user_id'] == user_id]
            for i in comp_user_ids:
                    comp_user_movies_rated = ratings[ratings['user_id'] == i]

                    # calculate jaccard similarity
                    jac = self.jaccard_similarity(set(user_movies_rated.movie_id), set(comp_user_movies_rated.movie_id))

                    # determine movie ids that both users have seen
                    both_seen_ids = set.intersection(*[set(user_movies_rated.movie_id), set(comp_user_movies_rated.movie_id)])

                    # for every movie that each of the two users have seen, average their ratings for that movie and
                    # store in parallel collections that will be used to calculate cosine similarity
                    user_scores = []
                    comp_scores = []
                    for j in both_seen_ids:
                        user_score = np.average(
                            ratings[(ratings['movie_id'] == j) & (ratings['user_id'] == user_id)].rating.values)
                        comp_score = np.average(
                            ratings[(ratings['movie_id'] == j) & (ratings['user_id'] == i)].rating.values)
                        user_scores.append(user_score)
                        comp_scores.append(comp_score)

                    # calculate cosine similarity score
                    cos = self.cosine_similarity(user_scores, comp_scores)
                    df_sim_scores.loc[len(df_sim_scores)] = [i, jac, cos, (jac + cos)]

            df_sim_scores[['id']] = df_sim_scores[['id']].astype(int)


            # sort by similarity scores
            df_sim_scores = df_sim_scores.sort_values(['comb'], ascending=[False])

            # the top recommended movies are the top rated movies by the most similar user that meet the following
            # conditions
            # 1) have not been seen by the target user
            # 2) have a minimum average rating of 3/5 stars
            #
            # We traverse through the list of users and do this for each of them in hopes of accumulating at least 3
            # recommendations
            target_movie_ids = []
            for i in df_sim_scores.id.values:
                filt = ratings[(ratings['user_id'] == i)]
                g = filt.groupby('movie_id').agg({'rating':[np.size, np.mean]})
                atleast_3 = g['rating']['mean'] >= 3
                g = g[atleast_3].sort_values([('rating', 'mean')], ascending=False)

                cands = g.reset_index().movie_id.values

                for c in cands:
                    if not(c in user_movies_rated.movie_id.values) and not(c in target_movie_ids):
                        target_movie_ids.append(c)

            # translate movie ids to movie titles
            recs = []
            for t in target_movie_ids:
                title = movies[(movies['id'] == t)].title.values[0]
                recs.append(title)
        return user_exists, recs[:self.MAX_RECS]

    def predict_rating(self, user_id, target_movie_id, sim_mode = 'comb', rttm = 0):

        # load data
        # done in-line with each web request.  Far from ideal and this pattern would never reach prod
        users, movies, ratings = self.load_data()
        user_exists = False
        all_user_ids = users.id.values[:]

        # ensure user requested exists in database
        # if not, bubble this info to resource layer which will throw the appropriate http error
        if user_id in all_user_ids:
            user_exists = True

            # get list of users who we want to generate similarity scores for
            comp_user_ids =  np.array(filter(lambda x: x != user_id, all_user_ids))

            # create a dataframe that keeps track of the similarity measures between all other users and the requested user
            df_sim_scores = pd.DataFrame(columns=['id', 'jac', 'cos', 'comb'])

            user_movies_rated = ratings[ratings['user_id'] == user_id]
            for i in comp_user_ids:
                comp_user_movies_rated = ratings[ratings['user_id'] == i]

                # calculate jaccard similarity
                jac = self.jaccard_similarity(set(user_movies_rated.movie_id), set(comp_user_movies_rated.movie_id))

                # determine movie ids that both users have seen
                both_seen_ids = set.intersection(*[set(user_movies_rated.movie_id), set(comp_user_movies_rated.movie_id)])

                # for every movie that each of the two users have seen, average their ratings for that movie and
                # store in parallel collections that will be used to calculate cosine similarity
                user_scores = []
                comp_scores = []
                for j in both_seen_ids:
                    user_score = np.average(
                        ratings[(ratings['movie_id'] == j) & (ratings['user_id'] == user_id)].rating.values)
                    comp_score = np.average(
                        ratings[(ratings['movie_id'] == j) & (ratings['user_id'] == i)].rating.values)
                    user_scores.append(user_score)
                    comp_scores.append(comp_score)

                # calculate cosine similarity score
                cos = self.cosine_similarity(user_scores, comp_scores)
                df_sim_scores.loc[len(df_sim_scores)] = [i, jac, cos, (jac + cos)]

            df_sim_scores[['id']] = df_sim_scores[['id']].astype(int)


            # sort by similarity scores
            df_sim_scores = df_sim_scores.sort_values([sim_mode], ascending=[False])

            numerator = 0.
            denominator = 0.
            sim_index = 0
            for i in df_sim_scores.id.values:
                score = np.average(
                    ratings[(ratings['movie_id'] == target_movie_id) & (ratings['user_id'] == i)].rating.values)

                if not isnan(score):
                    numerator = numerator + (10-sim_index) * score
                    denominator = denominator + (10 - sim_index)

                sim_index = sim_index + 1

            return (1-rttm) * numerator / denominator + rttm*np.average(ratings.rating.values)

    def generate_top_recommendation(self, target_user_id):
        users, movies, ratings = self.load_data()
        user_exists = False
        all_user_ids = users.id.values[:]

        # ensure user requested exists in database
        # if not, bubble this info to resource layer which will throw the appropriate http error
        if target_user_id in all_user_ids:
            user_exists = True

            movie_titles = movies.title.tolist()
            df = movies.join(ratings, on=['movie_id'], rsuffix='_r')
            del df['movie_id_r']
            rp = df.pivot_table(columns=['movie_id'], index=['user_id'], values='rating')
            rp = rp.fillna(0)

            Q = rp.values
            m, n = Q.shape

            # construct our weight matrix
            wt = Q > 0.5
            wt[wt == True] = 1
            wt[wt == False] = 0
            wt = wt.astype(np.float64, copy=False)

            # intitialize X and Y matrices
            X = self.MAX_RATING * np.random.rand(m, self.num_factors)
            Y = self.MAX_RATING * np.random.rand(self.num_factors, n)

            errors = []
            # rebuild X and Y, decreasing our error (hopefully) with each iteration
            for j in range(self.n_iterations):
                # Update Y
                for i, Wi in enumerate(wt.T):
                    Y[:, i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + self.lambda_ * np.eye(self.num_factors),
                                              np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
                # Update X
                for k, Wu in enumerate(wt):
                    X[k] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + self.lambda_ * np.eye(self.num_factors),
                                           np.dot(Y, np.dot(np.diag(Wu), Q[k].T))).T
                error = self.get_error(Q, X, Y, wt)
                errors.append(error)

            q_hat = np.dot(X, Y)

            # get rec for target user
            rec_movie, pred_score = self.gather_recommendations(wt, q_hat,m ,movie_titles, target_user_id)
            return user_exists, rec_movie

    def load_data(self):
        all_users = pd.read_csv('data/users.csv')
        all_movies = pd.read_csv('data/movies.csv')
        all_ratings = pd.read_csv('data/ratings.csv')
        return all_users, all_movies, all_ratings

    def square_rooted(self,x):
        return round(sqrt(sum([a * a for a in x])), 3)

    def cosine_similarity(self, r1, r2):
        num = sum(a * b for a, b in zip(r1, r2))
        denom = self.square_rooted(r1) * self.square_rooted(r2)
        return round(num / float(denom), 3)

    def jaccard_similarity(self, m1, m2):
        int_card = len(set.intersection(*[set(m1), set(m2)]))
        un_card = len(set.union(*[set(m1), set(m2)]))
        return round(int_card / float(un_card), 3)

    def get_error(self, Q, X, Y, W):
        return np.sum((W * (Q - np.dot(X, Y))) ** 2)

    def gather_recommendations(self, W, Q_hat,m, movie_titles, target_user_id):
        Q_hat -= np.min(Q_hat)
        Q_hat *= float(self.MAX_RATING) / np.max(Q_hat)
        movie_ids = np.argmax(Q_hat - self.MAX_RATING * W, axis=1)
        for jj, movie_id in zip(range(m), movie_ids):
            if(jj+1 == target_user_id):
                return movie_titles[movie_id], Q_hat[jj, movie_id]

        return -1,-1