# TODO: extend to find weak pairs in feature space between domains
# TODO: best model checkpoint
# TODO: proper batch indexing for uneven dataset sizes
# TODO: mkdir for LOG_DIR

import tensorflow as tf
import h5py
import image_utils as im
import numpy as np
import time
from numpy_utils import *
from model import *
from ops import *

""" setup and hyperparameters """
DATA_PATH = "../vrep_transfer_learning/datasets/d1.hdf5"
LAMBDA = 10.0
LR = 0.0002
BETA1 = 0.5
IM_OUTPUTS = 1
BATCH_SIZE = 1

MAX_EPOCHS = 1
IM_SUMMARY_PERIOD = 50
SAVE_PERIOD = 1000
TEST_PERIOD = 50

LOG_DIR = "log/a2b2a_regression/"

RESTORE = False

""" data """
print("Loading data resources")
data_time_start = time.time()

train_file = h5py.File(DATA_PATH, "r")
im_shape = (128, 128, 3)
train_images = train_file["images"]
train_images = im.h5py_to_array(train_images[:10000], im_shape)
a_train_images = train_images[:5000]
a_train_labels = train_file["joint_vel"][:].astype(np.float32)
a_dataset_size = a_train_images.shape[0]

tint_filter = [0.25, 0.5, 0.75]
b_train_images = im.tint_images(train_images[5000:], filter=tint_filter)
b_train_labels = a_train_labels
b_dataset_size = b_train_images.shape[0]

test_file = h5py.File("datasets/d1_test.hdf5", "r")
test_images = test_file["images"]
a_test_images = im.h5py_to_array(test_images, im_shape)
a_test_labels = test_file["joint_vel"][:].astype(np.float32)
b_test_images = im.tint_images(a_test_images, filter=tint_filter)
b_test_labels = a_test_labels

data_time_taken = time.time() - data_time_start
print("Data loaded, time taken: {:.5} seconds".format(data_time_taken))

""" global step """
global_step = tf.contrib.framework.get_or_create_global_step()
incr_global_step = tf.assign(global_step, global_step + 1)

""" model """
print("Building graph of model")
model_time_start = time.time()

a_image = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
a_label = tf.placeholder(dtype=tf.float32, shape=[None, 6])

a2b_image = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
a2b_label = tf.placeholder(dtype=tf.float32, shape=[None, 6])

a2b2a_image = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
a2b2a_label = tf.placeholder(dtype=tf.float32, shape=[None, 6])

b_image = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
b_label = tf.placeholder(dtype=tf.float32, shape=[None, 6])

b2a_image = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
b2a_label = tf.placeholder(dtype=tf.float32, shape=[None, 6])

b2a2b_image = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
b2a2b_label = tf.placeholder(dtype=tf.float32, shape=[None, 6])

test_image = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
test_label = tf.placeholder(dtype=tf.float32, shape=[None, 6])

with tf.variable_scope("a2b_gen"):
    a2b = generator(a_image)
with tf.variable_scope("b2a_gen"):
    b2a = generator(b_image)
with tf.variable_scope("b2a_gen", reuse=True):
    a2b2a = generator(a2b)
with tf.variable_scope("a2b_gen", reuse=True):
    b2a2b = generator(b2a)

with tf.variable_scope("a_discriminator") as scope:
    a_dis = discriminator(a_image)
    scope.reuse_variables()
    b2a_dis = discriminator(b2a)

with tf.variable_scope("b_discriminator") as scope:
    b_dis = discriminator(b_image)
    scope.reuse_variables()
    a2b_dis = discriminator(a2b)

with tf.variable_scope("regressor") as scope:
    a_feature = feature_extractor(a_image)
    a_reg = regressor_head(a_feature)
    scope.reuse_variables()
    a2b_feature = feature_extractor(a2b)
    a2b_reg = regressor_head(a2b_feature)
    a2b2a_feature = feature_extractor(a2b2a)
    a2b2a_reg = regressor_head(a2b2a_feature)

    feature_test = feature_extractor(test_image)
    reg_test = regressor_head(feature_test)

""" losses """
g_loss_a2b = l2_loss(a2b_dis, tf.ones_like(a2b_dis))
g_loss_b2a = l2_loss(b2a_dis, tf.ones_like(b2a_dis))
g_loss_cyc_a = l1_loss(a_image, a2b2a, weight=LAMBDA)
g_loss_cyc_b = l1_loss(b_image, b2a2b, weight=LAMBDA)
g_loss_sum = g_loss_a2b + g_loss_b2a + g_loss_cyc_a + g_loss_cyc_b

d_loss_a = l2_loss(a_dis, tf.ones_like(a_dis))
d_loss_b2a = l2_loss(b2a_dis, tf.zeros_like(b2a_dis))
d_loss_a_sum = d_loss_a + d_loss_b2a
d_loss_b = l2_loss(b_dis, tf.ones_like(b_dis))
d_loss_a2b = l2_loss(a2b_dis, tf.zeros_like(a2b_dis))
d_loss_b_sum = d_loss_b + d_loss_a2b

r_loss_a = l2_loss(a_reg, a_label)
r_loss_a2b = l2_loss(a2b_reg, a_label)
r_loss_a2b2a = l2_loss(a2b2a_reg, a_label)
r_loss_sum = r_loss_a + r_loss_a2b + r_loss_a2b2a

r_test_loss = l2_loss(reg_test, test_label)


t_var = tf.trainable_variables()
d_a_var = [var for var in t_var if "a_discriminator" in var.name]
d_b_var = [var for var in t_var if "b_discriminator" in var.name]
g_var = [var for var in t_var if "generator" in var.name]
r_var = [var for var in t_var if "regressor" in var.name]

d_a_train_op = tf.train.AdamOptimizer(LR, BETA1).minimize(d_loss_a_sum, global_step=None, var_list=d_a_var)
d_b_train_op = tf.train.AdamOptimizer(LR, BETA1).minimize(d_loss_b_sum, global_step=None, var_list=d_b_var)
g_train_op = tf.train.AdamOptimizer(LR, BETA1).minimize(g_loss_sum, global_step=None, var_list=g_var)
r_train_op = tf.train.AdamOptimizer(LR, BETA1).minimize(r_loss_sum, global_step=None, var_list=r_var)

model_time_taken = time.time() - model_time_start
print("Graph of model built, time taken: {:.5} seconds".format(model_time_taken))

""" summaries """
misc_time_start = time.time()
print("Loading summary operations and misc")

g_summaries_list = []
g_summaries_list.append(tf.summary.scalar("g_loss_a2b", g_loss_a2b))
g_summaries_list.append(tf.summary.scalar("g_loss_b2a", g_loss_b2a))
g_summaries_list.append(tf.summary.scalar("g_loss_cyc_a", g_loss_cyc_a))
g_summaries_list.append(tf.summary.scalar("g_loss_cyc_b", g_loss_cyc_b))
g_summaries_list.append(tf.summary.scalar("g_loss_sum", g_loss_sum))
g_summaries = tf.summary.merge(g_summaries_list)

d_summaries_list = []
d_summaries_list.append(tf.summary.scalar("d_loss_a", d_loss_a))
d_summaries_list.append(tf.summary.scalar("d_loss_b2a", d_loss_b2a))
d_summaries_list.append(tf.summary.scalar("d_loss_a_sum", d_loss_a_sum))
d_summaries_list.append(tf.summary.scalar("d_loss_b", d_loss_b))
d_summaries_list.append(tf.summary.scalar("d_loss_a2b", d_loss_a2b))
d_summaries_list.append(tf.summary.scalar("d_loss_b_sum", d_loss_b_sum))
d_summaries = tf.summary.merge(d_summaries_list)

r_summaries_list = []
r_summaries_list.append(tf.summary.scalar("r_loss_a", r_loss_a))
r_summaries_list.append(tf.summary.scalar("r_loss_a2b", r_loss_a2b))
r_summaries_list.append(tf.summary.scalar("r_loss_a2b2a", r_loss_a2b2a))
r_summaries_list.append(tf.summary.scalar("r_loss_sum", r_loss_sum))
r_summaries = tf.summary.merge(r_summaries_list)

r_test_a_summary = tf.summary.scalar("r_test_a", r_test_loss)
r_test_b_summary = tf.summary.scalar("r_test_b", r_test_loss)

im_summaries_list = []
im_summaries_list.append(tf.summary.image("a", a_image, max_outputs=IM_OUTPUTS))
im_summaries_list.append(tf.summary.image("a2b", a2b, max_outputs=IM_OUTPUTS))
im_summaries_list.append(tf.summary.image("a2b2a", a2b2a, max_outputs=IM_OUTPUTS))
im_summaries_list.append(tf.summary.image("b", b_image, max_outputs=IM_OUTPUTS))
im_summaries_list.append(tf.summary.image("b2a", b2a, max_outputs=IM_OUTPUTS))
im_summaries_list.append(tf.summary.image("b2a2b", b2a2b, max_outputs=IM_OUTPUTS))
im_summaries = tf.summary.merge(im_summaries_list)

""" collection """
tf.add_to_collection("reg_pred", a_reg)
tf.add_to_collection("reg_input", a_image)

misc_time_taken = time.time() - misc_time_start
print("Completed, time taken: {:.5} seconds".format(misc_time_taken))

""" initialise """
sess = tf.Session()
coord = tf.train.Coordinator()
summary_writer = tf.summary.FileWriter(LOG_DIR + "summaries/", sess.graph)

steps_per_epoch = np.int(np.ceil(float(a_dataset_size) / BATCH_SIZE))

""" save / restore """
saver = tf.train.Saver()
if RESTORE:
    saver.restore(sess, LOG_DIR + "model.ckpt")
else:
    sess.run(tf.global_variables_initializer())

""" train """
for epoch in range(1, MAX_EPOCHS + 1):
    a_shuffled_indices = RandomBatchIndices(a_dataset_size, batch_size=BATCH_SIZE)
    b_shuffled_indices = RandomBatchIndices(b_dataset_size, batch_size=BATCH_SIZE)
    for step_in_epoch in range(1, steps_per_epoch + 1):
        step_in_epoch_time_start = time.time()
        _, global_step_counter = sess.run([incr_global_step, global_step])

        fetches = {"d_a_train_op": d_a_train_op,
                   "d_b_train_op": d_b_train_op,
                   "g_train_op": g_train_op,
                   "r_train_op": r_train_op,
                   "a_reg": a_reg,
                   "a2b_reg": a2b_reg,
                   "a2b2a_reg": a2b2a_reg,
                   "g_summaries": g_summaries,
                   "d_summaries": d_summaries,
                   "r_summaries": r_summaries}

        a_batch_index = a_shuffled_indices.next_batch_index()
        b_batch_index = b_shuffled_indices.next_batch_index()
        feed_dict = {a_image: a_train_images[a_batch_index],
                     a_label: a_train_labels[a_batch_index],
                     b_image: b_train_images[b_batch_index]}

        # if (step_in_epoch == steps_per_epoch) and ((epoch % IM_SUMMARY_PERIOD) == 0):
        if global_step_counter % IM_SUMMARY_PERIOD == 0:
            fetches["im_summaries"] = im_summaries
            results = sess.run(fetches, feed_dict)
            summary_writer.add_summary(results["im_summaries"], global_step=global_step_counter)
        else:
            results = sess.run(fetches, feed_dict)

        summary_writer.add_summary(results["g_summaries"], global_step=global_step_counter)
        summary_writer.add_summary(results["d_summaries"], global_step=global_step_counter)
        summary_writer.add_summary(results["r_summaries"], global_step=global_step_counter)

        step_in_epoch_time_taken = time.time() - step_in_epoch_time_start
        print("Epoch {}/{}, step {}/{}, time: {:.5}s".format(epoch, MAX_EPOCHS, step_in_epoch, steps_per_epoch, step_in_epoch_time_taken))

        # if (step_in_epoch == steps_per_epoch) and ((epoch % TEST_PERIOD) == 0):
        if global_step_counter % TEST_PERIOD == 0:
            test_fetches = {"r_test_loss": r_test_loss,
                            "r_test_a_summary": r_test_a_summary}
            a_test_feed_dict = {test_image: a_test_images, test_label: a_test_labels}
            results = sess.run(test_fetches, a_test_feed_dict)
            summary_writer.add_summary(results["r_test_a_summary"], global_step=global_step_counter)

            test_fetches = {"r_test_loss": r_test_loss,
                            "r_test_b_summary": r_test_b_summary}
            b_test_feed_dict = {test_image: b_test_images, test_label: b_test_labels}
            results = sess.run(test_fetches, b_test_feed_dict)
            summary_writer.add_summary(results["r_test_b_summary"], global_step=global_step_counter)

        # if (step_in_epoch == steps_per_epoch) and ((epoch % SAVE_PERIOD) == 0):
        if global_step_counter % SAVE_PERIOD == 0:
            print("Saving model")
            saver.save(sess, LOG_DIR + "model.ckpt")

sess.close()

