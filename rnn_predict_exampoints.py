'''
@Author: Usman Ahmad

This program predicts possible exam points base on bonus points from the exercises.

Resources which I have used: 
Udemy Course: https://www.udemy.com/course/deep-learning-grundlagen-neuronale-netzwerke-mit-tensorflow/
Book: https://www.amazon.de/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291/ref=sr_1_2?__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=3823CVZITB8BP&keywords=hands+on+machine+learning+with+scikit-learn%2C+keras+and+tensorflow&qid=1573649195&sprefix=handson+machine+learning+%2Caps%2C193&sr=8-2

'''

from numpy import genfromtxt
import numpy as np
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt


''' Loading the Training Data:
This file consists of two columns: 
One Column: bonus points(Input Values)
Second Column: exam points as output values
Minimum Points: 0.0 = 0 Points
Maximum Points: 1.0 = 100 Points
'''

def fetch_training_data():
    dataset = genfromtxt('exam_bonus.csv', delimiter=',')
    
    # fetch the input and the expected output values, without the header
    X = dataset[1:len(dataset), 0]
    y = dataset[1:len(dataset), 1]
    return X, y

# Fetch batches from the training data
def next_batch(index, batch_size, n_inputs, n_features):
    x_batch = X_data[index:100 + index]
    y_batch = y_data[index:100 + index]
    x_batch = x_batch.reshape(batch_size, n_inputs, n_features)
    y_batch = y_batch.reshape(batch_size, n_inputs, n_features)

    return x_batch, y_batch


# These Hyperparameters were used 
learning_rate = 0.001
n_iterations = 100
batch_size = 5
n_neurons = 100

'''
n_inputs = 20 values(from 0.0 to 1.0 in 0.05 Steps)
'''
n_inputs= 20
n_features = 1 # Each input contains only one Feature(=Bonus Points)
n_outputs = 1 # = Exam Points

X = tf.placeholder(tf.float32, [None, n_inputs, n_features])
y = tf.placeholder(tf.float32, [None, n_inputs, n_outputs])

# In order to reduce the dimensionality of the output sequences to just one value, 
# OutputProjectionWrapper will be used
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu), output_size=n_outputs)


outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

X_data, y_data = fetch_training_data()

#Train the model
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):

        X_batch, y_batch = next_batch(n_inputs * iteration, batch_size, n_inputs, n_features)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
        print(iteration, "\Mean Squared Error:", mse)


    saver.save(sess, "./exam_bonus_model")

#Testing the Model
with tf.Session() as sess:
    saver.restore(sess, "./exam_bonus_model")
    X_new = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0]).reshape(1, 20, 1)
    y_pred = sess.run(outputs, feed_dict={X: X_new})
    
    # Print predictions
    print(y_pred)

    #Visualizing the results
    x_coordinates = X_new.reshape(n_inputs)
    y_coordinates = y_pred.reshape(n_inputs)

    plt.title("Predicting Exam Points based on Bonus Points", fontsize=14)
    plt.plot(x_coordinates[0:], y_coordinates[0:], "b.", markersize=10, label="prediction")
    plt.legend(loc="upper left")
    plt.xlabel("Bonus Points")
    plt.ylabel("Predicted Exam Points")
    plt.grid(True)
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.show()