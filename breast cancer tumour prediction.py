import pandas as pd
from sklearn import preprocessing, neighbors
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('Files/breast-cancer-wisconsin data.txt')  # import data file
df.replace('?', -99999,
           inplace=True)  # by substituting -99999, the algorithm considers
# those values as outliers and treats them accordingly
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

example = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
# supplying some data to predict the class of a patient. There are 2 IPs given bc we're asking about 2 patients.
# Note that they are unique orders, they cannot be similar to any case in the dataset


example = example.reshape(len(example), -1)
predict = clf.predict(example)
print(predict)
