
import tensorflow as tf
import numpy as np
from tensorflow import keras

# the relationship between the numbers in the given arrays is known to be y = 3x + 1.
# we will try to use tensorflow and neural networks to try to predict the value of y for any given x.

x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# create a simple neural network. It has 1 layer, and that layer has 1 neuron,
# the input shape to it is just 1 value.

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# compile the neural network specifying the optimizer and loss.
# sgd = stochastic gradient descent

model.compile(optimizer='sgd', loss='mean_squared_error')


# training the model with data(x) and result(y)

model.fit(x, y, epochs=500)

# predicting the result of a random data value after the model has been trained.

print(model.predict([12.0]))
