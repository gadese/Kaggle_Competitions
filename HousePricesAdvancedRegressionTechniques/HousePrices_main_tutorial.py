import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import math


# def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
#     """
#     Creates a list of random minibatches from (X, Y)
#
#     Arguments:
#     X -- input data, of shape (input size, number of examples)
#     Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
#     mini_batch_size -- size of the mini-batches, integer
#
#     Returns:
#     mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
#     """
#
#     m = X.shape[1]  # number of training examples
#     mini_batches = []
#
#     # Step 1: Shuffle (X, Y)
#     permutation = list(np.random.permutation(m))
#     shuffled_X = X[:, permutation]
#     shuffled_Y = Y[:, permutation].reshape((1, m))
#
#     # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
#     num_complete_minibatches = math.floor(
#         m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
#     for k in range(0, num_complete_minibatches):
#         ### START CODE HERE ### (approx. 2 lines)
#         mini_batch_X = shuffled_X[:, mini_batch_size * k:mini_batch_size * (k + 1)]
#         mini_batch_Y = shuffled_Y[:, mini_batch_size * k:mini_batch_size * (k + 1)]
#         ### END CODE HERE ###
#         mini_batch = (mini_batch_X, mini_batch_Y)
#         mini_batches.append(mini_batch)
#
#     # Handling the end case (last mini-batch < mini_batch_size)
#     if m % mini_batch_size != 0:
#         ### START CODE HERE ### (approx. 2 lines)
#         mini_batch_X = shuffled_X[:, (mini_batch_size * num_complete_minibatches):]
#         mini_batch_Y = shuffled_Y[:, (mini_batch_size * num_complete_minibatches):]
#         ### END CODE HERE ###
#         mini_batch = (mini_batch_X, mini_batch_Y)
#         mini_batches.append(mini_batch)
#
#     return mini_batches
#
# def create_placeholders(n_H0, n_W0, n_C0, n_y):
#     X = tf.placeholder(dtype=tf.float32, shape=(None, n_H0, n_W0, n_C0), name='X')
#     Y = tf.placeholder(dtype=tf.float32, shape=(None, n_y), name='Y')
#     return X, Y
#
#
# def initialize_parameters():
#     W1 = tf.get_variable("W1", [4, 4, 1, 1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
#     W2 = tf.get_variable("W2", [2, 2, 1, 1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
#
#     parameters = {"W1": W1,
#                   "W2": W2}
#     return parameters
#
#
# def forward_propagation(X, parameters):
#     # Retrieve the parameters from the dictionary "parameters"
#     W1 = parameters['W1']
#     W2 = parameters['W2']
#
#     # CONV2D: stride of 1, padding 'SAME'
#     Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
#     # RELU
#     A1 = tf.nn.relu(Z1)
#     # MAXPOOL: window 8x8, sride 8, padding 'SAME'
#     P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
#     # CONV2D: filters W2, stride 1, padding 'SAME'
#     Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
#     # RELU
#     A2 = tf.nn.relu(Z2)
#     # MAXPOOL: window 4x4, stride 4, padding 'SAME'
#     P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
#     # FLATTEN
#     P2 = tf.contrib.layers.flatten(P2)
#     # FULLY-CONNECTED without non-linear activation function (not not call softmax).
#     # 6 neurons in output layer.
#     Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)
#     return Z3
#
#
# def compute_cost(Z3, Y):
#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
#     return cost
#
#
# def model(X_train, Y_train, X_test, learning_rate=0.009,
#           num_epochs=100, minibatch_size=64, print_cost=True):
#     """
#     Implements a three-layer ConvNet in Tensorflow:
#     CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
#
#     Arguments:
#     X_train -- training set, of shape (None, 64, 64, 3)
#     Y_train -- test set, of shape (None, n_y = 6)
#     X_test -- training set, of shape (None, 64, 64, 3)
#     Y_test -- test set, of shape (None, n_y = 6)
#     learning_rate -- learning rate of the optimization
#     num_epochs -- number of epochs of the optimization loop
#     minibatch_size -- size of a minibatch
#     print_cost -- True to print the cost every 100 epochs
#
#     Returns:
#     train_accuracy -- real number, accuracy on the train set (X_train)
#     test_accuracy -- real number, testing accuracy on the test set (X_test)
#     parameters -- parameters learnt by the model. They can then be used to predict.
#     """
#
#     ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
#     (m, n_H0, n_W0, n_C0) = X_train.shape
#     n_y = Y_train.shape[1]
#     costs = []  # To keep track of the cost
#
#     # Create Placeholders of the correct shape
#     X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
#
#     # Initialize parameters
#     parameters = initialize_parameters()
#
#     # Forward propagation: Build the forward propagation in the tensorflow graph
#     Z3 = forward_propagation(X, parameters)
#
#     # Cost function: Add cost function to tensorflow graph
#     cost = compute_cost(Z3, Y)
#
#     # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
#     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
#     # Initialize all the variables globally
#     init = tf.global_variables_initializer()
#
#     # Start the session to compute the tensorflow graph
#     with tf.Session() as sess:
#
#         # Run the initialization
#         sess.run(init)
#
#         # Do the training loop
#         for epoch in range(num_epochs):
#
#             minibatch_cost = 0.
#             num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
#             minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
#
#             for minibatch in minibatches:
#                 # Select a minibatch
#                 (minibatch_X, minibatch_Y) = minibatch
#                 # IMPORTANT: The line that runs the graph on a minibatch.
#                 # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
#                 _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
#
#                 minibatch_cost += temp_cost / num_minibatches
#
#             # Print the cost every epoch
#             if print_cost == True and epoch % 5 == 0:
#                 print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
#             if print_cost == True and epoch % 1 == 0:
#                 costs.append(minibatch_cost)
#
#         # plot the cost
#         plt.plot(np.squeeze(costs))
#         plt.ylabel('cost')
#         plt.xlabel('iterations (per tens)')
#         plt.title("Learning rate =" + str(learning_rate))
#         plt.show()
#
#         # Calculate the correct predictions
#         predict_op = tf.argmax(Z3, 1)
#         correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
#
#         correct_prediction = sess.run([optimizer, cost], feed_dict={X: X_test})
#
#         return correct_prediction



num_features_pos = [1, 3, 4, 17, 18, 19, 20, 26, 34, 36, 37, 38, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 56, 59,
                    61, 62, 66, 67, 68, 69, 70, 71, 75, 76, 77]
features_one_hot = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33,
                    35, 39, 40, 41, 42, 53, 55, 57, 58, 60, 63, 64, 65, 72, 73, 74, 78, 79]

train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

sale_price = train_dataset.iloc[:, -1].values
train_features = train_dataset.iloc[:, :-1].values
test_features = test_dataset.iloc[:, :].values

X_train = np.zeros((train_features.shape[0], 1))
X_test = np.zeros((test_features.shape[0], 1))

my_imputer = SimpleImputer()
train_imputed = my_imputer.fit_transform(train_features[:, num_features_pos])
test_imputed = my_imputer.transform(test_features[:, num_features_pos])

for column in range(train_imputed.shape[1]):
    sc_X = StandardScaler()
    temp = sc_X.fit_transform(train_imputed[:, column].reshape(-1, 1).astype(float))
    temp_test = sc_X.transform(test_imputed[:, column].reshape(-1, 1).astype(float))
    X_train = np.append(X_train, temp, axis=1)
    X_test = np.append(X_test, temp_test, axis=1)

for column in features_one_hot:
    temp_train = pd.get_dummies(train_features[:, column])
    temp_test = pd.get_dummies(test_features[:, column])
    train_encoded, test_encoded = temp_train.align(temp_test, join='left', axis=1)
    X_train= np.append(X_train, train_encoded, axis=1)
    X_test = np.append(X_test, test_encoded, axis=1)

X_train = X_train[:, 1:]
X_test = X_test[:, 1:]

means =  np.mean(X_train[:, [54, 102, 103, 104, 113, 124, 126, 127, 128, 139, 142, 156, 206, 210, 222, 249, 263, 272]], axis= 0)
X_test[:, [54, 102, 103, 104, 113, 124, 126, 127, 128, 139, 142, 156, 206, 210, 222, 249, 263, 272]] = means

model = MLPRegressor(hidden_layer_sizes= (100,50, 25, 10, 5,), activation= 'relu', solver= 'adam', learning_rate ='adaptive', max_iter = 1000, learning_rate_init=0.01, alpha=0.01)
model.fit(X_train, np.ravel(sale_price))
predicted_test = model.predict(X_test)


# X_train = np.transpose(X_train)
# X_test = np.transpose(X_test)
#
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
# sale_price = sale_price.reshape(1, sale_price.shape[0], 1, 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)
#
# predicted_test = model(X_train, sale_price, X_test, num_epochs= 1)

output = pd.DataFrame({'Id': test_dataset['Id'], 'SalePrice': predicted_test})
output.to_csv('prediction.csv', index=False)



test = 1