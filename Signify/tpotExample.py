from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pandas as pd
import async_func


def get_target_and_factors(df, target_col):
	factors = df.loc[:, df.columns != target_col]
	target = df[target_col]
	return factors, target


def data_balancing(train_x, train_y):
	print('Data balancing start')
	sm = SMOTE(n_jobs=-1)
	een = EditedNearestNeighbours(n_jobs=-1)
	smen = SMOTEENN(smote=sm, enn=een)
	train_x_res, train_y_res = smen.fit_sample(train_x, train_y)
	print('Data balancing end')
	print(type(train_x_res))
	train_x_res = pd.DataFrame(train_x_res)
	train_y_res = pd.DataFrame(train_y_res)
	train_x_res.to_hdf('tpotbalanced.h5', 'training_x')
	train_y_res.to_hdf('tpotbalanced.h5', 'training_y', append=True)
	return train_x_res, train_y_res


datadir = async_func.get_data_dir('RunStatusCSV')
df_training = pd.read_csv(datadir + 'Data_C5_20180103-M4.csv')
df_training.append(pd.read_csv(datadir + 'Data_C5_20180116-M4.csv'))
df_testing = pd.read_csv(datadir + 'Data_C5_20180110-M4.csv')

df_testing.rename(columns={'Status': 'class'}, inplace=True)
df_training.rename(columns={'Status': 'class'}, inplace=True)

X_train, y_train = get_target_and_factors(df_training, 'class')
X_test, y_test = get_target_and_factors(df_testing, 'class')

X_train, y_train = data_balancing(X_train, y_train)

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
									random_state=42, verbosity=3, use_dask=True)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('phillips_pipeline2.py')
