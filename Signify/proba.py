import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.model_selection import train_test_split, ShuffleSplit
import async_func
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import cross_val_score


def plot_cm(cm, classes,
			normalize=False,
			title='Confusion matrix',
			cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()


def get_accuracy_and_confusion_matrix(trained_clf, classes, title, x_train, y_train, x_test, y_test):
	y_pred = trained_clf.predict(x_test)

	cnf_matrix = confusion_matrix(y_test, y_pred)
	np.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix
	plt.figure()
	plot_cm(cnf_matrix, classes=classes, title=title)
	plt.show()

	print('Accuracy test: {:.3f}'.format(trained_clf.score(x_test, y_test)))
	print('Accuracy training: {:.3f}'.format(trained_clf.score(x_train, y_train)))

	print('\n\n')


def data_balancing(train_x, train_y):
	print('Data balancing start')
	sm = SMOTE(n_jobs=-1)
	een = EditedNearestNeighbours(n_jobs=-1)
	smen = SMOTEENN(smote=sm, enn=een)

	train_x_res, train_y_res = smen.fit_sample(train_x, train_y)
	train_x_res = pd.DataFrame(data=train_x_res, columns=list(train_x.columns))
	train_y_res = pd.DataFrame(data=train_y_res, columns=['Status'])
	print('Data balancing end')

	train_x_res.to_hdf('balanced.h5', 'training_x')
	train_y_res.to_hdf('balanced.h5', 'training_y')
	return train_x_res, train_y_res


classes = ['Healthy', 'D1', 'D2', 'D3']
data_dir = async_func.get_data_dir('HugeCSVMulticlass') + 'Complete.csv'

df = pd.read_csv(data_dir)

df.drop('Time', axis='columns', inplace=True)

target = df.Status
cols = np.array(df.columns.tolist())
colsImp = ['DOS: atm_vocht', 'MEET: Licht RX', 'DOS: atm_druk', 'DOS: atm_temp', 'DOS: EPI_TEMP', 'lineNumber']
factors = df[colsImp]
x_train, x_test, y_train, y_test = train_test_split(factors, target, test_size=0.20, random_state=42)

rfc = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.7500000000000001,
							 min_samples_leaf=2, min_samples_split=12, n_estimators=100, n_jobs=-1)
# x_train_res, y_train_res = data_balancing(x_train, y_train)  # Resampled training sets

x_train_res = pd.read_hdf('balanced.h5', 'training_x')
y_train_res = pd.read_hdf('balanced.h5', 'training_y')

rfc.fit(x_train_res, y_train_res.values.ravel())

get_accuracy_and_confusion_matrix(rfc, classes, 'Confusion matrix, with resampled total data', x_train_res, y_train_res,
								  x_test, y_test)

colsImp = ['DOS: atm_vocht', 'MEET: Licht RX', 'DOS: atm_druk', 'DOS: atm_temp', ]
x_train_res = x_train_res[colsImp]


x_test = x_test[colsImp]

rfc = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.7500000000000001,
							 min_samples_leaf=2, min_samples_split=12, n_estimators=100, n_jobs=-1)
rfc.fit(x_train_res, y_train_res.values.ravel())
get_accuracy_and_confusion_matrix(rfc, classes, 'RandomForest with 4 variables', x_train_res, y_train_res, x_test,
								  y_test)

rfc = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.7500000000000001,
							 min_samples_leaf=2, min_samples_split=12, n_estimators=100, n_jobs=-1)
rfc.fit(x_train, y_train)
get_accuracy_and_confusion_matrix(rfc, classes, 'RandomForest with no resampling', x_train, y_train, x_test,
								  y_test)
