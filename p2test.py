
import tensorflow as tf
import numpy as np
from tensorflow import keras
import statistics
import math
import random



numpy_x = np.random.randn(100000)  # generates a pseudo-random normal distribution

# training data sets x and y.
x = []
y = []
for el in numpy_x:   # converting the numpy return type to a python usable type
    x.append(el)

mu = statistics.mean(x)
sigma = statistics.stdev(x)
upper_bound = mu + sigma
lower_bound = mu - sigma
print(mu, sigma, upper_bound, lower_bound)

# for normal distribution, f(x) = (1/sigma*sqrt(2pi))*(e^-(1/2)*((x-mu)/sigma)^2)

lower_limit = (1 / (sigma * math.sqrt(6.28))) * pow(2.71, (-((lower_bound - sigma) ** 2) / (2 * sigma ** 2)))
upper_limit = (1 / (sigma * math.sqrt(6.28))) * pow(2.71, (-((upper_bound - sigma) ** 2) / (2 * sigma ** 2)))
for i in range(len(x)):
    value = (1 / (sigma * math.sqrt(6.28))) * pow(2.71, (-((x[i] - sigma) ** 2) / (2 * sigma ** 2)))
    y.append(value)

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])


# compile the neural network specifying the optimizer and loss.
# sgd = stochastic gradient descent

model.compile(optimizer='sgd', loss='mean_squared_error')

# training the model with data(x) and result(y)

x_train = x[0:9000]
y_train = y[0:9000]

test_set = x[9001:9998]
print(len(x))
print(len(x_train))
print(len(y_train))
print(len(test_set))

model.fit(x_train, y_train, epochs=5)

count = 0

for i in test_set:
    output = model.predict([i])
    if output > upper_limit or output < lower_limit:
        #print("software getting degraded")
        count += 1
for j in test_set:
    output = model.predict([j])

    if output > upper_limit or output < lower_limit:
        count += 1
        if count >= 50:
            print("Your upgrade is slow. Refactoring may be a good idea.")
            break
else:
    print("Execution speed for the upgrade is OK")




# # monitoring the performance of the software over time.
# success_test = np.random.randn(10)  # generating fake run times to be used for testing.
# failure_test = []
# for i in range(1):  # using values surely out of 1 stdev from the mean
#     failure_test.append(i)
# count = 0
#
# for i in range(len(success_test)):
#     output = model.predict([success_test[i]])
#     if output > upper_limit or output < lower_limit:
#         #print("software getting degraded")
#         count += 1
# for j in range(len(failure_test)):
#     output = model.predict([failure_test[j]])
#
#     if output > upper_limit or output < lower_limit:
#         count += 1
#         if count >= 3:
#             print("Your upgrade is slow. Refactoring may be a good idea.")
#             break
# else:
#     print("Execution speed for the upgrade is OK")

