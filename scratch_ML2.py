import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def sigmoid(z):
    return 1.0/(1 + np.exp(-z))


def loss_f(target_value, hypo_value):
    loss = -np.mean(target_value * (np.log(hypo_value)) - (1 - target_value) * np.log(1 - hypo_value))
    return loss


# gradient descent
# a = a - lr * da
# b = b - lr * db
# where, da is the partial derivative of the Loss function with respect to a
# and db is the partial derivative of the Loss function with respect to b
# da = (1/m)*(hypo_value — target_value).X
# db = (1/m)*(hypo_value - target_value)

def gradients(X, target_value, hypo_value):
    # X --> Input.
    # a --> weights (parameter).
    # b --> bias (parameter).
    # m-> number of training examples.

    m = X.shape[0]

    # Gradient of loss w.r.t weights.N:B X.T is the transpose of X
    da = (1 / m) * np.dot(X.T, (hypo_value - target_value))

    # Gradient of loss w.r.t bias.
    db = (1 / m) * np.sum((hypo_value - target_value))

    return da, db


def normalize(X):
    # X --> Input.

    # m-> number of training examples
    # n-> number of features
    m, n = X.shape

    # Normalizing all the n features of X.
    for i in range(n):
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    return X


def train(X, y, bs, epochs, lr):
    # X --> Input.
    # y --> true/target value.
    # bs --> Batch Size.
    # epochs --> Number of iterations.
    # lr --> Learning rate.

    # m-> number of training examples
    # n-> number of features
    m, n = X.shape

    # Initializing weights and bias to zeros.
    global a, b
    a = np.zeros((n, 1))
    b = 0

    # Reshaping y.
    y = y.reshape(m, 1)

    # Normalizing the inputs.
    x = normalize(X)

    # Empty list to store losses.
    losses = []

    # Training loop.
    for epoch in range(epochs):
        for i in range((m - 1) // bs + 1):
            # Defining batches. SGD.
            start_i = i * bs
            end_i = start_i + bs
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]

            # Calculating hypothesis/prediction.
            hypo_value = sigmoid(np.dot(xb, a) + b)

            # Getting the gradients of loss w.r.t parameters.
            da, db = gradients(xb, yb, hypo_value)

            # Updating the parameters.
            a -= lr * da
            b -= lr * db

        # Calculating loss and appending it in the list.
        l = loss_f(y, sigmoid(np.dot(X, a) + b))
        losses.append(l)

    # returning weights, bias and losses(List).
    return a, b, losses


def predict(X):
    # X --> Input.

    # Normalizing the inputs.
    x = normalize(X)

    # Calculating predictions/y_hat.
    preds = sigmoid(np.dot(X, a) + b)

    # Empty List to store predictions.
    pred_class = []
    # if y_hat >= 0.5 --> round up to 1
    # if y_hat < 0.5 --> round down to 0
    pred_class = [1 if i >= 0.5 else 0 for i in preds]

    return np.array(pred_class)


def accuracy(target, hypo):
    accuracy = np.sum(target == hypo) / len(target)
    return accuracy


dataset = pd.read_csv('C:\\Users\\Tanifor\\Desktop\\ML\\project_2\\dataset\\data.txt', delimiter=',')
#print(dataset.head(4))

# differentiating the dependent variable y from the independent variable X
X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
y = dataset.iloc[:, 8].values

# splitting the dataset into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

a, b, losses = train(X_train, y_train, bs=100, epochs=1000, lr=0.0001)
predicted = predict(X_test)
df = pd.DataFrame({'target values': y_test, 'predicted values': predicted})
print(df)
print('accuracy:', accuracy(y_test, predicted))
