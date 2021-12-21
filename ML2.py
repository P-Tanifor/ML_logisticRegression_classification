import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('C:\\Users\\Tanifor\\Desktop\\ML\\project_2\\dataset\\data.txt', delimiter=',')
#print(dataset.head(4))

# differentiating the dependent variable y from the independent variable X
X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
y = dataset.iloc[:, 8].values

# splitting the dataset into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# feature scaling i.e. normalize the data within a particular range
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# training the model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# predicting the values for the test set
y_pred = classifier.predict(X_test)
#print(y_pred)

# checking the accuracy
accuracy = accuracy_score(y_test, y_pred)
#print('Accuracy:', accuracy)

# comparing real values and the predicted values

df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
print(df)
print('Accuracy:', accuracy)