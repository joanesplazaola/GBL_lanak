import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import async_func
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import itertools
import glob
import os


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confussudion matrix',
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


def get_target_and_factors(df, target_col):
    factors = df.loc[:, df.columns != target_col]
    target = df[target_col]
    return factors, target


def jaja(clf, X_train, y_train, X_test, y_test, classes):
    clf = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print(
        'Random forest test acc: ' + str(clf.score(X_test, y_test)) + " train Acc: " + str(clf.score(X_train, y_train)))
    plot_confusion_matrix(confusion_matrix(y_test, predictions), classes=classes, title='Random Forest')

    predictions = clf.predict(X_train)
    plot_confusion_matrix(confusion_matrix(y_train, predictions), classes=classes, title='Random Forest')


def get_all_run_info():
    files = glob.glob(async_func.get_data_dir('RunStatusCSV') + '*.csv')
    for file in files:
        df = pd.read_csv(file)
        path, filename = os.path.split(file)

        factors, target = get_target_and_factors(df, 'Status')
        X_train, X_test, y_train, y_test = train_test_split(factors, target)

        rfc = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1, max_depth=None,
                                     min_samples_split=50)
        jaja(rfc, X_train, y_train, X_test, y_test, ['Healthy', 'D1', 'D2', 'D3'])


if __name__ == '__main__':
    datadir = async_func.get_data_dir('RunStatusCSV')
    df_training = pd.read_csv(datadir + 'Data_C5_20180103-M4.csv')
    df_training.append(pd.read_csv(datadir + 'Data_C5_20180116-M4.csv'))
    df_testing = pd.read_csv(datadir + 'Data_C5_20180110-M4.csv')

    train_factors, train_target = get_target_and_factors(df_training, 'Status')
    test_factors, test_target = get_target_and_factors(df_testing, 'Status')
    rfc = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1, max_depth=None, min_samples_split=50)

    jaja(rfc, train_factors, train_target, test_factors, test_target, ['Healthy', 'D1', 'D2', 'D3'])

    rfc = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1, max_depth=None, min_samples_split=50)
    factors, target = get_target_and_factors(df_testing, 'Status')

    X_train, X_test, y_train, y_test = train_test_split(factors, target, )
    jaja(rfc, X_train, y_train, X_test, y_test, ['Healthy', 'D1', 'D2', 'D3'])
