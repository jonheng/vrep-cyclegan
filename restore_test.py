import tensorflow as tf
import numpy as np
import time

sess = tf.Session()
print ("Loading model...")
restore_time_start = time.time()
saver = tf.train.import_meta_graph("log/a2b2a_regression/model.ckpt.meta")
saver.restore(sess, "log/a2b2a_regression/model.ckpt")
restore_time_taken = time.time() - restore_time_start
print("Model loaded, time taken: {}".format(restore_time_taken))
pred_op = tf.get_collection("reg_pred")[0]
input_op = tf.get_collection("reg_input")[0]

test_image = np.zeros(shape=[1, 128, 128, 3], dtype=np.float32)
jointvel = sess.run(pred_op, feed_dict={input_op: test_image})
print jointvel
