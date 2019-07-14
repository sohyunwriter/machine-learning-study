import tensorflow as tf
import numpy as np
import pandas as pd

tf.set_random_seed(777)

## data preprocessing
dic = {'asymptomatic': 0, 'nonanginal': 1, 'nontypical': 2, 'typical': 3}
dic2 = {'fixed': 1, 'normal': 2, 'reversable': 3}
dic3 = {'Yes': 0, 'No': 1}

def cate(list, dic):
    for i, k in enumerate(list):
        list[i] = dic[k]
    return list

df = pd.read_csv("./data/Heart_Train.csv")

df.ix[df.AHD == "No", "AHD"] = 0
df.ix[df.AHD == "Yes", "AHD"] = 1
df.ix[df.ChestPain == "asymptomatic", "ChestPain"] = 0
df.ix[df.ChestPain == "nonanginal", "ChestPain"] = 1
df.ix[df.ChestPain == "nontypical", "ChestPain"] = 2
df.ix[df.ChestPain == "typical", "ChestPain"] = 3
df.ix[df.Thal == "fixed", "Thal"] = 0
df.ix[df.Thal == "normal", "Thal"] = 1
df.ix[df.Thal == "reversable", "Thal"] = 2

df.head()
train_x = df.iloc[:, 0:-1]
train_y = df.iloc[:, [-1]]
num_feature = len(df.columns) - 1

train_x = train_x.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

df = pd.read_csv("./data/Heart_Test.csv")
test_x = df.iloc[:, 0:-1]
test_y = df.iloc[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, num_feature])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([num_feature, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis using sigmoid
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = tf.reduce_mean(Y*tf.log(hypothesis) + (1 - Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

pred = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed = {X: train_x, Y: train_y}
    for step in range(10001):
        sess.run(train, feed_dict=feed)
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict=feed))

    h, c, a = sess.run([hypothesis, pred, accuracy], feed_dict=feed)
    print("\nHypothesis: ", h, "\nCorrect (Y) ", c, "\nAccuracy: ", a)


