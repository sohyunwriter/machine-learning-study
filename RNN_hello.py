import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()  # session

'''
# one hot encoding vector 'helo'
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]
'''

# 12-1
'''
# One cell RNN input_dim(4) -> output_dim(2)
hidden_size = 2
# cell 만들기
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size) # rnn/LSTM cell A

x_data = np.array([[[1, 0, 0, 0]]], dtype = np.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32) # output, hidden state

sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())
'''

'''
# One cell RNN input_dim (4) -> output_dim (2), sequence : 5
x_data = np.array([[h, e, l, l, o]], dtype = np.float32)
# print(x_data)

hidden_size = 2
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32) # output, hidden state

sess.run(tf.global_variables_initializer())
print(outputs.eval())
#pp.pprint(outputs.eval())
'''
'''
# One cell RNN input_dim (4) -> output_dim (2), sequence : 5, batch 3
# 3 batches 'hello', 'eolll', 'lleel'
x_data = np.array([[h, e, l, l, o], [e, o, l, l, l], [l, l, e, e, l]], dtype = np.float32)
# print(x_data)

hidden_size = 2
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32) # output, hidden state

sess.run(tf.global_variables_initializer())
print(outputs.eval())
print(x_data)
print(outputs.shape)

# (batch_size, sequence_length, hidden_size)
'''
# 12-2
h = [1, 0, 0, 0, 0]   # 0
i = [0, 1, 0, 0, 0]   # 1
e = [0, 0, 1, 0, 0]   # 2
l = [0, 0, 0, 1, 0]   # 3
o = [0, 0, 0, 0, 1]   # 4
idx2char = ['h', 'i', 'e', 'l', 'o']

# RNN model
hidden_size = 5  # output from the LSTM
input_dim = 5 # one-hot vector size
batch_size = 1 # one sentence
sequence_length = 6 # |ihello| == 6

x_data = [[0, 1, 0, 2, 3, 3]] # input
x_one_hot = [[h, i, h, e, l, l]] # one-hot

y_data = [[1, 0, 2, 3, 3, 4]] # y

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim]) # X - one hot (batch_size, sequence_length, input_dim)
Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y Label (batch_size, sequence_length)

cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden_size) # return as tuple(not tensor)
#cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state = initial_state, dtype = tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_outputs=6, activation_fn = None) # output print??

outputs = tf.reshape(outputs, [batch_size, sequence_length, 6])
weights = tf.ones([batch_size, sequence_length]) # weight??
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss) # loss

train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss) # train

prediction = tf.argmax(outputs, axis=2) # result

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000): # epochs
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        #print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str : ", ''.join(result_str))
