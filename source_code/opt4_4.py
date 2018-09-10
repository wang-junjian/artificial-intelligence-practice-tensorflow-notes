import tensorflow as tf


w = tf.Variable(tf.constant(5, dtype=tf.float32))
loss = tf.square(w+1)
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	
	for _ in range(40):
		sess.run(train_step)
		print('loss value: ', sess.run(loss), ' w value: ', sess.run(w))

