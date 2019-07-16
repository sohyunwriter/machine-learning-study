/**
    ML STUDY 1(19.07.16) - logistic regression for heart dataset
    @sohyunwriter (brightcattle@gmail.com)
**/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

df = pd.read_csv("./data/Heart_Train.csv")
df.head()

## 1. data preprocessing - categorical value --> numeric
# creating dummy variable
a = pd.get_dummies(df['ChestPain'], prefix = "cp")
b = pd.get_dummies(df['Thal'], prefix = "thal")
c = pd.get_dummies(df['Slope'], prefix = "slope")
frames = [df, a, b, c] ## ???
df = pd.concat(frames, axis=1)
df.head()
df = df.drop(columns = ['ChestPain', 'Thal', 'Slope'])
df.ix[df.AHD == "No", "AHD"] = 0
df.ix[df.AHD == "Yes", "AHD"] = 1
y_train = df.iloc[:, [-1]] # y_train
x_data = df.iloc[:, 0:-1]
x_train = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values  # normalize data (all columns of X) # x_train

df = pd.read_csv("./data/Heart_Test.csv")
# creating dummy variable
a = pd.get_dummies(df['ChestPain'], prefix = "cp")
b = pd.get_dummies(df['Thal'], prefix = "thal")
c = pd.get_dummies(df['Slope'], prefix = "slope")
frames = [df, a, b, c]
df = pd.concat(frames, axis=1)
df.head()
df = df.drop(columns = ['ChestPain', 'Thal', 'Slope'])
df.ix[df.AHD == "No", "AHD"] = 0
df.ix[df.AHD == "Yes", "AHD"] = 1
y_test = df.iloc[:, [-1]] # y_test
x_data = df.iloc[:, 0:-1]

# normalize data
x_test = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values # x_test

## 2. creating model for logistic regression using tensorflow
num_feature = len(x_train.columns)
X = tf.placeholder(tf.float32, shape=[None, num_feature])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([num_feature, 1], mean= 0.01, stddev = 0.01), name='weight') # xavier initialzation
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis using sigmoid
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis)) # cross entropy
train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)  # learning late = 0.001

pred = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_train, Y: y_train})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_train, Y: y_train})) # print cost each 200 step
            train_accuracy = sess.run(accuracy, feed_dict={X: x_train, Y: y_train})
            test_accuracy = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
            print("Train Accuracy : ", train_accuracy, "Test Accuracy : ", test_accuracy) # print train_accuracy, test_accuracy each 200 step

    # print final train_accuracy, test_accuracy
    train_accuracy = sess.run(accuracy, feed_dict={X: x_train, Y: y_train})
    #print("\nHypothesis: ", h, "\nCorrect (Y) ", c, "\nAccuracy: ", a)
    test_accuracy = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
    print("Final Train Accuracy : ", train_accuracy, "Final Test Accuracy : ", test_accuracy)
    
    
## result
/**
0 0.349779
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
200 0.32993463
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
400 0.3147886
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
600 0.30298212
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
800 0.29360494
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
1000 0.28603125
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
1200 0.27982104
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
1400 0.27465826
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
1600 0.27031162
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
1800 0.26660895
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
2000 0.2634202
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
2200 0.26064575
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
2400 0.2582082
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
2600 0.25604704
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
2800 0.2541142
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
3000 0.25237134
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
3200 0.2507876
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
3400 0.2493377
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
3600 0.24800117
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
3800 0.24676104
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
4000 0.2456034
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
4200 0.24451646
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
4400 0.24349052
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
4600 0.24251737
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
4800 0.24159004
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
5000 0.24070267
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
5200 0.2398503
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
5400 0.23902857
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
5600 0.2382339
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
5800 0.2374631
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
6000 0.23671347
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
6200 0.23598269
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
6400 0.23526876
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
6600 0.23456985
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
6800 0.23388448
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
7000 0.23321128
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
7200 0.23254907
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
7400 0.23189688
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
7600 0.23125379
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
7800 0.230619
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
8000 0.22999181
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
8200 0.22937162
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
8400 0.22875793
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
8600 0.22815016
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
8800 0.22754799
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
9000 0.22695099
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
9200 0.22635883
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
9400 0.22577122
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
9600 0.22518791
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
9800 0.22460864
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
10000 0.2240332
Train Accuracy :  0.9276596 Test Accuracy :  0.9298246
Final Train Accuracy :  0.9276596 Final Test Accuracy :  0.9298246
**/

