def read_data_1m():
	print('reading movielens 1m data')
	ratings = pd.read_csv('../data/ml-1m/ratings.dat', 
		sep='::',
		names=[
			'userId',
			'movieId',
			'rating',
			'ts'
		],
		engine='python')
	movies = pd.read_csv('../data/ml-1m/movies.dat', 
		sep='::',
		names=[
			'movieId',
			'title',
			'genres'
		],
		engine='python')
	users = pd.read_csv('../data/ml-1m/users.dat', 
		sep='::', 
		names = [
			'userId',
			'gender',
			'age',
			'occupation',
			'zip'
		],
		engine='python')
	logs = ratings.join(movies, on='movieId', how='left', rsuffix='_movie')
	logs = logs.join(users, on='userId', how='left', rsuffix='_movie')
	return logs

def process_title():
	pass

def process_genres():
	pass



def preprocess_movie_data_1m(logs, min_number_of_reviews=1000):
	print('preparing ratings log')
	# remove ratings of movies with < N ratings. too few ratings will cause the recsys to get stuck in offline evaluation
	movies_to_keep = pd.DataFrame(logs.movieId.value_counts())\
		.loc[pd.DataFrame(logs.movieId.value_counts())['movieId']>=min_number_of_reviews].index
	logs = logs.loc[logs['movieId'].isin(movies_to_keep)]
	# shuffle rows to deibas order of user ids
	logs = logs.sample(frac=1)
	# create a 't' column to represent time steps for the bandit to simulate a live learning scenario
	logs['t'] = np.arange(len(logs))
	logs.index = logs['t']
	logs['liked'] = logs['rating'].apply(lambda x: 1 if x >= 4.5 else 0)
	return logs
    

def get_ratings_1m(min_number_of_reviews=1000):
	logs = read_data_1m()
	logs = preprocess_movie_data_1m(logs, min_number_of_reviews=min_number_of_reviews)
	return logs