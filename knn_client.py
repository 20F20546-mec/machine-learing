# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import classification_report

# read dataset using pandas
df = pd.read_csv('bank-additional-full.csv', delimiter=';')

df = df[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan','campaign', 'pdays', 'previous', 'poutcome', 'y']]
df.info()

# pre-process categorical data
objfeatures = df.select_dtypes(include="object").columns
le = preprocessing.LabelEncoder()

# tranforms features
for feat in objfeatures:
    df[feat] = le.fit_transform(df[feat].astype(str))

X = df.drop('y', axis=1)
y = df['y']

# normalization
X = preprocessing.StandardScaler().fit_transform(X.astype(int))

# splitting into training and testing tests
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# knn is run twice
# the first time is using 'Euclidean Distance' to find optimum K
# the second time is using 'Manhattan Distance' to find optimum K

# euclidean distance
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(weights='distance', n_neighbors=i).fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# plot the minimum error
plt.figure(figsize=(10,6))
plt.plot(range(1, 40), error_rate,color='black', linestyle='dashed',
         marker='o',markerfacecolor='yellow', markersize=6)
plt.title('Error Rate vs. K Value with Distance Metric: Euclidean')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:", min(error_rate), "at K =", error_rate.index(min(error_rate)))

# euclidean distance
acc = []
for i in range(1, 40):
    neigh = KNeighborsClassifier(weights='distance', n_neighbors=i).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat))

# plot the maximum accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), acc, color='black', linestyle='dashed',
         marker='o', markerfacecolor='yellow', markersize=6)
plt.title('Accuracy vs. K Value with Distance Metric: Euclidean')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-", max(acc), "at K =", acc.index(max(acc)))

# performance metrics with euclidean distance
classifier = KNeighborsClassifier(weights='distance', n_neighbors=i).fit(X_train, y_train)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(f"Model Accuracy : {accuracy_score(y_pred,y_test)*100:.2f}%")
print(f"Model F1-Score : {f1_score(y_pred,y_test,average='weighted')*100:.2f}%")
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5, scoring="recall")
print("Cross Val Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Cross Val Standard Deviation: {:.2f} %".format(accuracies.std()*100))
print(classification_report(y_pred, y_test,zero_division=1))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# manhattan distance
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(weights='distance', n_neighbors=i, p=1).fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# plot the minimum error
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate,color='black', linestyle='dashed',
         marker='o',markerfacecolor='yellow', markersize=6)
plt.title('Error Rate vs. K Value with Distance Metric: Manhattan')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:", min(error_rate), "at K =", error_rate.index(min(error_rate)))

# manhattan distance
acc = []
for i in range(1, 40):
    neigh = KNeighborsClassifier(weights='distance', n_neighbors=i, p=1).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat))

# plot the maximum accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), acc, color='black', linestyle='dashed',
         marker='o', markerfacecolor='yellow', markersize=6)
plt.title('Accuracy vs. K Value with Distance Metric: Manhattan')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-", max(acc), "at K =", acc.index(max(acc)))

# performance metrics with manhattan distance
classifier = KNeighborsClassifier(weights='distance', n_neighbors=i, p=1).fit(X_train, y_train)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(f"Model Accuracy : {accuracy_score(y_pred,y_test)*100:.2f}%")
print(f"Model F1-Score : {f1_score(y_pred,y_test,average='weighted')*100:.2f}%")
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5, scoring="recall")
print("Cross Val Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Cross Val Standard Deviation: {:.2f} %".format(accuracies.std()*100))
print(classification_report(y_pred, y_test,zero_division=1))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))