import tensorflow as tf
import numpy as np

sess = tf.Session()
saver = tf.train.import_meta_graph("log/regressor/model.ckpt.meta")
saver.restore(sess, "log/regressor/model.ckpt")
pred_op = tf.get_collection("pred")[0]
input_op = tf.get_collection("image_batch")[0]

test_image = np.zeros(shape=[1, 128, 128, 3], dtype=np.float32)
jointvel = sess.run(pred_op, feed_dict={input_op: test_image})
print jointvel
