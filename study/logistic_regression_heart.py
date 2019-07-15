import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

tf.set_random_seed(777)

## data preprocessing

def data_clean(df):
    df.head()
    for feature_name in ['AHD', 'ChestPain', 'Thal']:
        #pd.get_dummies(df[feature_name], prefix=feature_name, drop_first=True)
        df = pd.concat([df.drop(feature_name, axis=1), pd.get_dummies(df[feature_name])], axis=1)

    for feature_name in ['Age', 'RestBP', 'Chol', 'RestECG', 'MaxHR', 'Oldpeak', 'Slope','Ca']:
        #std = np.std(df[feature_name])
        #mean = np.mean(df[feature_name])
        max = np.max(df[feature_name])
        min = np.max(df[feature_name])
        #df[feature_name] = (df[feature_name] - mean) / std
        df[feature_name] = (df[feature_name] - min) / max - min
        #mX = mean(df[feature_name], 1)
        #print("000")

    train_x = df.iloc[:, 0:-1]
    #train_x = train_x.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    train_y = df.iloc[:, [-1]]
    return train_x, train_y

df = pd.read_csv("./data/Heart_Train.csv")
train_x, train_y = data_clean(df)
#print(train_x, train_y)
num_feature = len(train_x.columns)

df = pd.read_csv("./data/Heart_Test.csv")
test_x, test_y = data_clean(df)

X = tf.placeholder(tf.float32, shape=[None, num_feature])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([num_feature, 1], mean= 0.01, stddev = 0.01), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis using sigmoid
eps = 1e-8
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=hypothesis, labels=Y)
#cost = -tf.reduce_mean(Y*tf.log(hypothesis + eps) + (1-Y)*tf.log(1-hypothesis + eps))
#cost = tf.reduce_mean(Y*tf.log(tf.clip_by_value(hypothesis, eps, 1.)) + (1 - Y) * tf.log(tf.clip_by_value(1-hypothesis, eps, 1.)))  # add eps to prevent divide by zero
train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

pred = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: train_x, Y: train_y})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: train_x, Y: train_y}))
            train_accuracy = sess.run(accuracy, feed_dict={X: train_x, Y: train_y})
            test_accuracy = sess.run(accuracy, feed_dict={X: test_x, Y: test_y})
            print("Train Accuracy : ", train_accuracy, "Test Accuracy : ", test_accuracy)

    train_accuracy = sess.run(accuracy, feed_dict={X: train_x, Y: train_y})
   # print("\nHypothesis: ", h, "\nCorrect (Y) ", c, "\nAccuracy: ", a)
    test_accuracy = sess.run(accuracy, feed_dict={X: test_x, Y: test_y})
    print("Train Accuracy : ", train_accuracy, "Test Accuracy : ", test_accuracy)


    # Train Accuracy :  0.6255319 Test Accuracy :  0.5964912
