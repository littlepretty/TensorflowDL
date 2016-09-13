import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""Data"""
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""Model"""
# x is N by 784, represent N images
x = tf.placeholder(tf.float32, [None, 28*28])
W = tf.Variable(tf.zeros([28*28, 10]))
b = tf.Variable(tf.zeros([10]))

# y is N by 10, softmax of 10 classes
y = tf.nn.softmax(tf.matmul(x, W) + b)

"""Train"""
# y prime are the label of N imgaes
y_ = tf.placeholder(tf.float32, [None, 10])
# reduced_sum = N by 1, the cross entropy of N images
# reduced_mean is the mean of the above N entropies
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                              reduction_indices=[1]))
# minimize mean cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # training use y_ instead of y!
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

"""Prediction"""
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
result = sess.run(accuracy,
                  feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print result
