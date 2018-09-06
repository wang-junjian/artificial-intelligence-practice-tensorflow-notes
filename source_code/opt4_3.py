import numpy as np
import tensorflow as tf

BATCH_SIZE = 8
SEED = 23455
PROFIT = 1
COST = 9

rdm = np.random.RandomState(SEED)

X = rdm.rand(32, 2)
Y_ = [[x0+x1+(rdm.rand()/10.0-0.05)] for (x0,x1) in X]

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal((2,1), stddev=1, seed=1))

y = tf.matmul(x, w1)

loss = tf.reduce_sum(tf.where(tf.greater(y, y_), COST*(y-y_), PROFIT*(y_-y))) 
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)

	STEPS = 20000 
	for i in range(STEPS):
		start = i*BATCH_SIZE % 32
		end = start+BATCH_SIZE
		sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
		if (i+1) % 500 == 0:
			print(i, ' w1: ', sess.run(w1))
