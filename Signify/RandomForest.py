#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 11:42:16 2018

@author: ezugasti
"""
import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import tree
from detect_peaks import detect_peaks as dp
import itertools
from sklearn.ensemble import AdaBoostClassifier


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


import async_func

data_dir = async_func.get_data_dir('HugeCSV') + 'Complete.csv'

df = pd.read_csv(data_dir)
y = df.Status
cols = np.array(df.columns.tolist())
X = df[cols[0:-1]]

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.66)

clf = RandomForestClassifier(n_estimators=60, max_depth=None, min_samples_split=2)
clf = clf.fit(train_x, train_y)
predictions = clf.predict(test_x)
trainAcc = accuracy_score(train_y, clf.predict(train_x))
testAcc = accuracy_score(test_y, predictions)
print(testAcc)

list1 = clf.feature_importances_
list2 = X.columns.tolist()
kk = [list(x) for x in zip(*sorted(zip(list1, list2), key=lambda pair: pair[0]))]
plt.bar(np.arange(len(kk[0])), kk[0][::-1])
plt.xticks(np.arange(len(kk[0])), kk[1][::-1], rotation=90)
# tree.export_graphviz(clf,out_file='PhilipsCompleteTree.dot')
deriv = np.abs(np.diff(kk[0][::-1]))
indexes = dp(deriv, mph=deriv.max() / 6, mpd=0)

for i in indexes:
	cols = kk[1][::-1][0:i + 1]
	Xn = X[cols]
	train_x, test_x, train_y, test_y = train_test_split(Xn, y, test_size=0.66)
	clf = RandomForestClassifier(n_estimators=60, max_depth=None, min_samples_split=2)
	clf = clf.fit(train_x, train_y)
	predictions = clf.predict(test_x)
	trainAcc = accuracy_score(train_y, clf.predict(train_x))
	testAcc = accuracy_score(test_y, predictions)
	print(testAcc)
	plot_confusion_matrix(confusion_matrix(test_y, predictions), classes=['0', '1'])

for i in indexes:
	cols = kk[1][::-1][0:i + 1]
	Xn = X[cols]
	train_x, test_x, train_y, test_y = train_test_split(Xn, y, test_size=0.66)
	clf = AdaBoostClassifier(n_estimators=100)
	clf.fit(train_x, train_y)
	predictions = clf.predict(test_x)
	trainAcc = accuracy_score(train_y, clf.predict(train_x))
	testAcc = accuracy_score(test_y, predictions)
	print(testAcc)
	plot_confusion_matrix(confusion_matrix(test_y, predictions), classes=['0', '1'])
