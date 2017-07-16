import tensorflow as tf
import h5py
import image_utils as im
import numpy as np
from model import *

""" setup and hyperparameters """
data_path = "datasets/d1_test.hdf5"
batch_size = 16

""" data """
file = h5py.File(data_path, "r")
images = file["images"]
im_shape = (128, 128, 3)
dataset_size = images.shape[0]
a_real_np = im.h5py_to_array(images, im_shape)
a_real_labels = file["joint_vel"][:].astype(np.float32)
a_real_batch_op, a_real_labels_batch_op = tf.train.shuffle_batch([a_real_np, a_real_labels],
                                                                 batch_size=batch_size,
                                                                 capacity=500,
                                                                 min_after_dequeue=50,
                                                                 num_threads=2,
                                                                 enqueue_many=True,
                                                                 allow_smaller_final_batch=True)

""" model """
with tf.variable_scope("regressor"):
    a_feature = feature_extractor(a_real_batch_op)
    pred = regressor_head(a_feature)

global_vars = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(0.001)

""" inference """
inference_input = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
inference_label = tf.placeholder(dtype=tf.float32, shape=[None, 6])
with tf.variable_scope("regressor", reuse=True):
    infer_feature = feature_extractor(inference_input)
    infer_pred = regressor_head(infer_feature)
infer_loss = tf.reduce_mean(tf.square(inference_label - infer_pred))

""" loss """
mse = tf.reduce_mean(tf.square(a_real_labels_batch_op - pred))
grads = optimizer.compute_gradients(mse, var_list=global_vars)
train_op = optimizer.apply_gradients(grads)

""" execution """
sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
steps_per_epoch = np.int(np.ceil(1.0 * dataset_size / batch_size))

sess.run(tf.global_variables_initializer())
for i in range(100):
    # a_real_batch, a_real_labels_batch = sess.run([a_real_batch_op, a_real_labels_batch_op])
    # print("Step {}, batch shapes: {}, {}".format(i, a_real_batch.shape, a_real_labels_batch.shape))
    _, mse_eval = sess.run([train_op, mse])
    print "MSE: ", mse_eval