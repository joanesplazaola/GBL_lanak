#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:34:23 2018

@author: ezugasti
"""

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
	GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from detect_peaks import detect_peaks as dp
import itertools
from tqdm import tqdm

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
# Import the visualizer
from yellowbrick.features import RadViz
from sklearn import preprocessing
from yellowbrick.features.importances import FeatureImportances
import async_func


def plot_confusion_matrix(cm, classes,
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

	print(cm)
	plt.figure()
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

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show(block=True)


def get_col_indexes(cols, train_x):
	index_dict = dict()
	for col in cols:
		index_dict[col] = train_x.columns.get_loc(col)
	return index_dict


# define two outlier detection tools to be compared
cart = DecisionTreeClassifier()
num_trees = 100
classifiers = {
	#    "Bagging": BaggingClassifier(base_estimator=cart, n_estimators=num_trees),
	"ExtraTrees": ExtraTreesClassifier(n_estimators=num_trees, min_samples_split=2),
	#    "AdaBoost": AdaBoostClassifier(n_estimators=num_trees),
	#    "StochasticGradientBoosting": GradientBoostingClassifier(n_estimators=num_trees),
	"RandomForest": RandomForestClassifier(n_estimators=num_trees, max_depth=None, min_samples_split=2)}

datadir = async_func.get_data_dir('HugeCSVMulticlass')
df = pd.read_csv(datadir + 'Complete.csv')
df = df.drop('Time', axis='columns')
line = df.lineNumber
yy = df.Status
df = df.loc[:, df.columns != 'Status']

cols = np.array(df.columns.tolist())

XX = df[cols]
X = (XX)
y = yy

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.10)
print(get_col_indexes(cols, train_x))
print('balancing start')
##Balancing Train data
# sm = SMOTE(n_jobs=-1)
# een = EditedNearestNeighbours(n_jobs=-1)
# smen = SMOTEENN(smote=sm, enn=een)
# train_x_res, train_y_res = smen.fit_sample(train_x, train_y)
print('balancing stop')

kk = []

for i, (clf_name, clf) in tqdm(enumerate(classifiers.items())):
	clf = clf.fit(train_x, train_y)
	predictions = clf.predict(test_x)
	kk.append(clf)
	trainAcc = accuracy_score(train_y, clf.predict(train_x))
	testAcc = accuracy_score(test_y, predictions)
	print(clf_name + ' test acc: ' + str(testAcc) + " train Acc: " + str(trainAcc))
	plot_confusion_matrix(confusion_matrix(test_y, predictions), classes=['Healthy', 'D1', 'D2', 'D3'], title=clf_name)

# plot feature importances
for i, (clf_name, clf) in tqdm(enumerate(classifiers.items())):
	fig = plt.figure()
	ax = fig.add_subplot()

	viz = FeatureImportances(clf, ax=ax)
	viz.fit(X, y)
	viz.poof()

colsImp = ['DOS: atm_vocht', 'MEET: Licht RX', 'DOS: atm_druk', 'DOS: atm_temp', 'DOS: EPI_TEMP']
for i, (clf_name, clf) in tqdm(enumerate(classifiers.items())):
	nf = 7
	cols = colsImp
	#    Xn=X[cols]

	train_xn = train_x[cols]
	test_xn = test_x[cols]
	train_yn = train_y
	test_yn = test_y
	clf.fit(train_xn, train_yn)
	predictions = clf.predict(test_xn)
	trainAcc = accuracy_score(train_yn, clf.predict(train_xn))
	testAcc = accuracy_score(test_yn, predictions)
	print(
		clf_name + ' with ' + str(len(cols)) + ' Features: test acc: ' + str(testAcc) + " train Acc: " + str(trainAcc))
	plot_confusion_matrix(confusion_matrix(test_y, predictions), classes=['Healthy', 'D1', 'D2', 'D3'], title=clf_name)
