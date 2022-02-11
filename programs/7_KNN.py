from sklearn import datasets,preprocessing,neighbors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
%matplotlib inline



iris=load_iris()
print("Iris data")
print(iris)
print("\n")
print("Iris feature_names")
print("\n")
print(iris.feature_names)
print("\n")
print("integers representing features(0=setosa,1=versicolor,2=virginica)")
print("\n")
print(iris.target)
print("\n")
print("3classes of target")
print("\n")

print(iris.target_names)
print("\n")

print("total of 150 observations and 4 features")
print("\n")
print(iris.data.shape)
print("\n")


X, y = iris.data[:, :], iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 0, train_size = 0.7)
print("shape of train and test objects")
print("\n")

print(X_train.shape)
print(X_test.shape)
print("\n")

print("shape of new y objects")
print("\n")
print(y_train.shape)
print(y_test.shape)
print("\n")

print("training data before preprocessing")
print(X_train)
print("\n")
#Normalization
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
print("display scaled data")
print("\n")
print(X_train)
print("\n")
X_test = scaler.transform(X_test)



scores = []
k_range = range(1,15)
for k in k_range:

  knn = neighbors.KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  
  
print("Prediction of Species: {}".format(y_pred))  
print("Accuracy score")
print(accuracy_score(y_test, y_pred))
print("Confusion matrix")
print(confusion_matrix(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))
