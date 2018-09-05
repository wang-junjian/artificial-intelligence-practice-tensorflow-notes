#coding:utf-8

import numpy as np
import tensorflow as tf

SEED = 23455
rng = np.random.RandomState(SEED)

#生成样本集
X = rng.rand(32, 2)
Y = [[int(x0+x1<1)] for (x0, x1) in X]

#定义神经网络
x = tf.placeholder(tf.float32, shape=(None,2))
y_ = tf.placeholder(tf.float32, shape=(None,1))

w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)

	for i in range(10000):
		sess.run(train_step, feed_dict={x:X, y_:Y})
		if i % 500 == 0:
			print(sess.run(loss, feed_dict={x:X, y_:Y}))

	print(sess.run(w1))
	print(sess.run(w2))

