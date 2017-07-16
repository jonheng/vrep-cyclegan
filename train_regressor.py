import tensorflow as tf
import h5py
import image_utils as im
import numpy as np
from model import *
from numpy_utils import *

""" setup and hyperparameters """
DATA_PATH = "../vrep_transfer_learning/datasets/d1.hdf5"
BATCH_SIZE = 64
TOTAL_EPOCHS = 50
SAVE_PERIOD = 10
TEST_PERIOD = 1
LOG_DIR = "log/regressor3/"
RESTORE = False

""" data """
train_file = h5py.File(DATA_PATH, "r")
im_shape = (128, 128, 3)
train_images = train_file["images"]
dataset_size = train_images.shape[0]
train_images = im.h5py_to_array(train_images, im_shape)
train_labels = train_file["joint_vel"][:].astype(np.float32)

test_file = h5py.File("datasets/d1_test.hdf5", "r")
test_images = test_file["images"]
test_images = im.h5py_to_array(test_images, im_shape)
test_labels = test_file["joint_vel"][:].astype(np.float32)

image_batch = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
label_batch = tf.placeholder(dtype=tf.float32, shape=[None, 6])

""" model """
with tf.variable_scope("regressor"):
    a_feature = feature_extractor(image_batch)
    pred = regressor_head(a_feature)

global_vars = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(0.001)

""" step counters """
global_step = tf.contrib.framework.get_or_create_global_step()
incr_global_step = tf.assign(global_step, global_step + 1)

""" loss """
mse = tf.reduce_mean(tf.square(label_batch - pred))
grads = optimizer.compute_gradients(mse, var_list=global_vars)
# Increment global step every time a train is done
with tf.control_dependencies([incr_global_step]):
    train_op = optimizer.apply_gradients(grads)

# epoch_mse = tf.constant(0.0, dtype=tf.float32)
# epoch_mse_reset = tf.assign(epoch_mse, 0)
# epoch_mse_update = tf.assign(epoch_mse, epoch_mse)

""" summaries """
loss_summaries = tf.summary.scalar("train_mse_loss_epoch", mse)
test_summaries = tf.summary.scalar("test_mse_loss_epoch", mse)

""" collection """
tf.add_to_collection("pred", pred)
tf.add_to_collection("image_batch", image_batch)

""" execution """
sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
steps_per_epoch = np.int(np.ceil(float(dataset_size) / BATCH_SIZE))
summary_writer = tf.summary.FileWriter(LOG_DIR + "summaries/", sess.graph)

""" save / restore """
saver = tf.train.Saver()
if RESTORE:
    saver.restore(sess, LOG_DIR + "model.ckpt")
else:
    sess.run(tf.global_variables_initializer())

for epoch in range(1, TOTAL_EPOCHS + 1):
    shuffled_indices = RandomBatchIndices(dataset_size, batch_size=BATCH_SIZE)
    for step_in_epoch in range(steps_per_epoch):
        batch_index = shuffled_indices.next_batch_index()
        _, mse_eval, step, train_summary_str = sess.run([train_op, mse, global_step, loss_summaries],
                                                        feed_dict={image_batch: train_images[batch_index],
                                                                   label_batch: train_labels[batch_index]})
        print("Epoch {}, {}/{}, train_mse_loss {}".format(epoch, step_in_epoch, steps_per_epoch, mse_eval))

        # At end of epoch
        if (step_in_epoch + 1) == steps_per_epoch:
            if epoch % TEST_PERIOD == 0:
                test_loss, test_summary_str = sess.run([mse, test_summaries],
                                                       feed_dict={image_batch: test_images,
                                                                  label_batch: test_labels})
                print("test_mse_loss {}\n".format(test_loss))
                summary_writer.add_summary(test_summary_str, epoch)
                summary_writer.add_summary(train_summary_str, epoch)  #TODO: implement a proper average for train summary

            if epoch % SAVE_PERIOD == 0:
                print "Saving model"
                saver.save(sess, LOG_DIR + "model.ckpt")