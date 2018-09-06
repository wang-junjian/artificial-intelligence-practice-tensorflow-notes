# Tensorflow FAQ

* 1 [没有声明为变量(tf.Variable)，导致的错误。](http://nbviewer.jupyter.org/github/wang-junjian/artificial-intelligence-practice-tensorflow-notes/blob/master/faq/faq001.ipynb)
```py
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-9-2a9e1f27ff10> in <module>()
19 
20 loss = tf.reduce_mean(tf.square(y-y_))
---> 21 train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
22 
23 with tf.Session() as sess:

/usr/local/lib/python3.6/site-packages/tensorflow/python/training/optimizer.py in minimize(self, loss, global_step, var_list, gate_gradients, aggregation_method, colocate_gradients_with_ops, name, grad_loss)
404           "No gradients provided for any variable, check your graph for ops"
405           " that do not support gradients, between variables %s and loss %s." %
--> 406           ([str(v) for _, v in grads_and_vars], loss))
407 
408     return self.apply_gradients(grads_and_vars, global_step=global_step,

ValueError: No gradients provided for any variable, check your graph for ops that do not support gradients, between variables ["<tf.Variable 'Variable:0' shape=(2, 1) dtype=float32_ref>", "<tf.Variable 'Variable_1:0' shape=(2, 1) dtype=float32_ref>", "<tf.Variable 'Variable_2:0' shape=(2, 1) dtype=float32_ref>", "<tf.Variable 'Variable_3:0' shape=(2, 1) dtype=float32_ref>"] and loss Tensor("Mean_6:0", shape=(), dtype=float32).
```
