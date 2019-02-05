import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import ShuffleSplit, cross_val_score

import async_func
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import itertools


def get_target_and_factors(df, target_col):
    return df.loc[:, df.columns != target_col], df[target_col]


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
    train_y_res.to_hdf('balanced.h5', 'training_y', append=True)
    return train_x_res, train_y_res


'''
datadir = async_func.get_data_dir('HugeCSVMulticlass')

df = pd.read_csv(datadir + 'Complete.csv')

df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d')
df = df.set_index(df.Time)
df = df.sort_index()

# training_data = df[(df.Time.dt.month == 9) | (df.Time.dt.month == 11)]
testing_data = df[df.Time.dt.month == 10]

# training_data.drop(columns=['Time'], inplace=True)
testing_data.drop(columns=['Time'], inplace=True)

# x_train, y_train = get_target_and_factors(training_data, 'Status')
x_test, y_test = get_target_and_factors(testing_data, 'Status')
'''
x_test = pd.read_hdf('balanced.h5', 'test_x')
y_test = pd.read_hdf('balanced.h5', 'test_y')
# train_x_res, train_y_res = data_balancing(x_train, y_train)
train_x_res = pd.read_hdf('balanced.h5', 'training_x')
train_y_res = pd.read_hdf('balanced.h5', 'training_y')
rfc = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1, max_depth=None, min_samples_split=50)

cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
scores = cross_val_score(rfc, train_x_res, train_y_res, cv=cv)
print("Cross validation:\n\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

rfc = rfc.fit(train_x_res, train_y_res.values.ravel())
predictions = rfc.predict(x_test)

print('Random forest test acc: ' + str(rfc.score(x_test, y_test)) + " train Acc: " + str(
    rfc.score(train_x_res, train_y_res)))
plot_confusion_matrix(confusion_matrix(y_test, predictions), classes=['Healthy', 'D1', 'D2', 'D3'],
                      title='Random Forest')

predictions = rfc.predict(train_x_res)
plot_confusion_matrix(confusion_matrix(train_y_res, predictions), classes=['Healthy', 'D1', 'D2', 'D3'],
                      title='Random Forest')
