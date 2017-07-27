# Trains a neural network to perform regression on joint velocity labels for a 3dof arm
# Testing how performance and losses change with size of dataset available

import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import image_utils as im
from ops import *
from model import *
from PIL import Image

""" HYPER-PARAMETERS """
NUM_EPISODES = 20
TOTAL_IMAGES = NUM_EPISODES * 16
BATCH_SIZE = 1
NUM_EPOCHS = 1
MAX_TRAINING_IMAGES = TOTAL_IMAGES * NUM_EPOCHS
TOTAL_STEPS = MAX_TRAINING_IMAGES / BATCH_SIZE
RESTORE = False
NUM_THREADS = 2
IMG_SHAPE = [128, 128, 3]
NUM_JOINTS = 3
IM_OUTPUTS = 1

# A_TRAIN_DIR = ["./datasets/3dof-arm-grid/", "./datasets/3dof-arm/"]
A_TRAIN_DIR = ["./datasets/3dof-arm-test/"]
B_TRAIN_DIR = "./datasets/3dof-arm-test/"
TEST_DIR = "./datasets/3dof-arm-test/"
LOG_DIR = "./log/a2b2a_3dof/"

TEST_PERIOD = 250
SAVE_PERIOD = 10000
IM_SUMMARY_PERIOD = 10

LAMBDA = 10
LR = 0.0002
BETA1 = 0.5

if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)


def preprocess_image(image_tensor):
    # Converts images to a range of [-1, 1]
    image = (tf.cast(image_tensor, dtype=tf.float32) / 127.5) - 1
    return image


def read_labels(label_path, size=None):
    """
    Input: Label file path
    Returns: List of labels 
    """
    file = open(label_path, "r")
    data = file.readlines()
    if size is None:
        size = len(data)
    output = np.empty(shape=(size, 3), dtype=np.float32)
    for i in range(size):
        line_array = np.array(data[i].rstrip().split(" "), dtype=np.float32)
        output[i, :] = line_array[1:4]

    return output


def read_images(image_paths):
    """
    Input: Image directory 
    Returns: Numpy array of images
    """
    size = len(image_paths)
    image_array = np.empty(shape=[size, ] + IMG_SHAPE, dtype=np.float32)
    counter = 0
    for image in image_paths:
        img = Image.open(image)
        arr = np.array(img)
        arr = arr.astype(np.float32) / 127.5 - 1
        image_array[counter] = arr
        counter += 1
    return image_array

""" Dataset A processing (source dataset with labels) """
a_train_images_dir = []
a_train_labels_path = []
a_train_images_paths = []
for dir in A_TRAIN_DIR:
    a_train_images_dir.append(dir + "images/")
    a_train_labels_path.append(dir + "joint_vel.txt")
for dir in a_train_images_dir:
    a_train_images_paths += [dir + "image" + str(i) + ".jpg" for i in range(1, len(os.listdir(dir)) + 1)]
a_train_labels_array = np.empty(shape=(0, 3), dtype=np.float32)
for label_path in a_train_labels_path:
    a_train_labels_array = np.concatenate([a_train_labels_array, read_labels(label_path)])
print "Dataset A size: ", len(a_train_images_paths)
print "Dataset labels shape: ", a_train_labels_array.shape
a_train_images = tf.convert_to_tensor(a_train_images_paths, dtype=tf.string)
a_train_labels = tf.convert_to_tensor(a_train_labels_array, dtype=tf.float32)
a_train_input_queue = tf.train.slice_input_producer([a_train_images, a_train_labels],
                                                    num_epochs=NUM_EPOCHS,
                                                    shuffle=True)
a_train_image = tf.image.decode_jpeg(tf.read_file(a_train_input_queue[0]))
a_train_image = preprocess_image(a_train_image)
a_train_image.set_shape(IMG_SHAPE)
a_train_label = a_train_input_queue[1]
a_train_image_batch, a_train_label_batch = tf.train.batch([a_train_image, a_train_label],
                                                          batch_size=BATCH_SIZE,
                                                          allow_smaller_final_batch=True,
                                                          num_threads=NUM_THREADS)

""" Dataset B processing (target dataset without labels) """
b_train_images_dir = B_TRAIN_DIR + "images/"
# b_train_images_paths = [b_train_images_dir + str(i) + ".jpg" for i in range(1, len(os.listdir(b_train_images_dir)) + 1)]
b_train_images_paths = []
for path in os.listdir(b_train_images_dir):
    b_train_images_paths.append(b_train_images_dir + path)
b_train_images = tf.convert_to_tensor(b_train_images_paths, dtype=tf.string)
b_train_input_queue = tf.train.slice_input_producer([b_train_images],
                                                    shuffle=True)
b_train_image = tf.image.decode_jpeg(tf.read_file(b_train_input_queue[0]))
b_train_image = preprocess_image(b_train_image)
b_train_image.set_shape(IMG_SHAPE)
b_train_image_batch = tf.train.batch([b_train_image],
                                     batch_size=BATCH_SIZE,
                                     allow_smaller_final_batch=True,
                                     num_threads=NUM_THREADS)


""" Train dataset processing """
test_images_dir = TEST_DIR + "images/"
test_labels_path = TEST_DIR + "joint_vel.txt"
test_size = len(os.listdir(test_images_dir))
print("Test size: {}".format(test_size))
test_images_paths = [test_images_dir + "image" + str(i) + ".jpg" for i in range(1, test_size + 1)]
test_images_array = read_images(test_images_paths)
test_labels_array = read_labels(test_labels_path, test_size)
a_test_images = test_images_array
a_test_labels = test_labels_array
b_test_images = im.tint_images(a_test_images, [0.25, 0.5, 0.75])
b_test_labels = test_labels_array


""" model """
test_images = tf.placeholder(dtype=tf.float32, shape=[None, ] + IMG_SHAPE)
test_labels = tf.placeholder(dtype=tf.float32, shape=[None, NUM_JOINTS])
a = tf.placeholder(dtype=tf.float32, shape=[None, ] + IMG_SHAPE)
a_label = tf.placeholder(dtype=tf.float32, shape=[None, NUM_JOINTS])
b = tf.placeholder(dtype=tf.float32, shape=[None, ] + IMG_SHAPE)

with tf.variable_scope("a2b_gen"):
    a2b = generator(a)
with tf.variable_scope("b2a_gen"):
    b2a = generator(b)
with tf.variable_scope("b2a_gen", reuse=True):
    a2b2a = generator(a2b)
with tf.variable_scope("a2b_gen", reuse=True):
    b2a2b = generator(b2a)

with tf.variable_scope("a_discriminator") as scope:
    a_dis = discriminator(a)
    scope.reuse_variables()
    b2a_dis = discriminator(b2a)

with tf.variable_scope("b_discriminator") as scope:
    b_dis = discriminator(b)
    scope.reuse_variables()
    a2b_dis = discriminator(a2b)

with tf.variable_scope("regressor") as scope:
    a_feature = feature_extractor(a)
    a_pred = regressor_head_3dof(a_feature)
    scope.reuse_variables()
    a2b_feature = feature_extractor(a2b)
    a2b_pred = regressor_head_3dof(a2b_feature)
    a2b2a_feature = feature_extractor(a2b2a)
    a2b2a_pred = regressor_head_3dof(a2b2a_feature)
    test_feature = feature_extractor(test_images)
    test_pred = regressor_head_3dof(test_feature)

""" step counters """
global_step = tf.contrib.framework.get_or_create_global_step()
incr_global_step = tf.assign(global_step, global_step + 1)

""" loss """
g_loss_a2b = l2_loss(a2b_dis, tf.ones_like(a2b_dis))
g_loss_b2a = l2_loss(b2a_dis, tf.ones_like(b2a_dis))
g_loss_cyc_a = l1_loss(a, a2b2a, weight=LAMBDA)
g_loss_cyc_b = l1_loss(b, b2a2b, weight=LAMBDA)
g_loss_sum = g_loss_a2b + g_loss_b2a + g_loss_cyc_a + g_loss_cyc_b

d_loss_a = l2_loss(a_dis, tf.ones_like(a_dis))
d_loss_b2a = l2_loss(b2a_dis, tf.zeros_like(b2a_dis))
d_loss_a_sum = d_loss_a + d_loss_b2a
d_loss_b = l2_loss(b_dis, tf.ones_like(b_dis))
d_loss_a2b = l2_loss(a2b_dis, tf.zeros_like(a2b_dis))
d_loss_b_sum = d_loss_b + d_loss_a2b

r_loss_a = l2_loss(a_pred, a_label)
r_loss_a2b = l2_loss(a2b_pred, a_label)
r_loss_a2b2a = l2_loss(a2b2a_pred, a_label)
r_loss_sum = r_loss_a + r_loss_a2b + r_loss_a2b2a

r_loss_test = l2_loss(test_pred, test_labels)

t_var = tf.trainable_variables()
d_a_var = [var for var in t_var if "a_discriminator" in var.name]
d_b_var = [var for var in t_var if "b_discriminator" in var.name]
g_var = [var for var in t_var if "generator" in var.name]
r_var = [var for var in t_var if "regressor" in var.name]

d_a_train_op = tf.train.AdamOptimizer(LR, BETA1).minimize(d_loss_a_sum, global_step=None, var_list=d_a_var)
d_b_train_op = tf.train.AdamOptimizer(LR, BETA1).minimize(d_loss_b_sum, global_step=None, var_list=d_b_var)
g_train_op = tf.train.AdamOptimizer(LR, BETA1).minimize(g_loss_sum, global_step=None, var_list=g_var)
r_train_op = tf.train.AdamOptimizer(LR, BETA1).minimize(r_loss_sum, global_step=None, var_list=r_var)

""" summaries """
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

r_test_a_summary = tf.summary.scalar("r_test_a", r_loss_test)
r_test_b_summary = tf.summary.scalar("r_test_b", r_loss_test)

im_summaries_list = []
im_summaries_list.append(tf.summary.image("a", a, max_outputs=IM_OUTPUTS))
im_summaries_list.append(tf.summary.image("a2b", a2b, max_outputs=IM_OUTPUTS))
im_summaries_list.append(tf.summary.image("a2b2a", a2b2a, max_outputs=IM_OUTPUTS))
im_summaries_list.append(tf.summary.image("b", b, max_outputs=IM_OUTPUTS))
im_summaries_list.append(tf.summary.image("b2a", b2a, max_outputs=IM_OUTPUTS))
im_summaries_list.append(tf.summary.image("b2a2b", b2a2b, max_outputs=IM_OUTPUTS))
im_summaries = tf.summary.merge(im_summaries_list)

""" collection """
tf.add_to_collection("pred", test_pred)
tf.add_to_collection("image", test_images)

# """ textfile logs """
# train_textlog = open(LOG_DIR + "train_loss.txt", "w")
# test_textlog = open(LOG_DIR + "test_loss.txt", "w")

""" execution """
sess = tf.Session()

""" save / restore """
saver = tf.train.Saver()
# r_saver = tf.train.Saver(r_var)
if RESTORE:
    saver.restore(sess, LOG_DIR + "model.ckpt")
else:
    sess.run(tf.global_variables_initializer())

# This is important for slice_input_producer num_epochs to work
sess.run(tf.local_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

summary_writer = tf.summary.FileWriter(LOG_DIR + "summaries/", sess.graph)

for step in range(TOTAL_STEPS):
    step_timer = time.time()
    _, global_step_counter = sess.run([incr_global_step, global_step])
    a_batch, a_label_batch, b_batch = sess.run([a_train_image_batch, a_train_label_batch, b_train_image_batch])
    b_batch = im.tint_images(b_batch, [0.25, 0.5, 0.75])
    fetches = {"d_a_train_op": d_a_train_op,
               "d_b_train_op": d_b_train_op,
               "g_train_op": g_train_op,
               "r_train_op": r_train_op,
               "g_summaries": g_summaries,
               "d_summaries": d_summaries,
               "r_summaries": r_summaries,
               "a_pred": a_pred,
               "a2b_pred": a2b_pred,
               "a2b2a_pred": a2b2a_pred}
    feed_dict = {a: a_batch,
                 a_label: a_label_batch,
                 b: b_batch}

    if (step + 1) % IM_SUMMARY_PERIOD == 0:
        fetches["im_summaries"] = im_summaries
        results = sess.run(fetches, feed_dict)
        summary_writer.add_summary(results["im_summaries"], global_step=global_step_counter)
    else:
        results = sess.run(fetches, feed_dict)

    summary_writer.add_summary(results["g_summaries"], global_step=global_step_counter)
    summary_writer.add_summary(results["d_summaries"], global_step=global_step_counter)
    summary_writer.add_summary(results["r_summaries"], global_step=global_step_counter)

    if (step + 1) % TEST_PERIOD == 0 or (step + 1) == TOTAL_STEPS:
        a_test_fetches = {"r_loss_test": r_loss_test,
                          "r_test_a_summary": r_test_a_summary}
        a_test_feed_dict = {test_images: a_test_images,
                            test_labels: a_test_labels}
        results = sess.run(a_test_fetches, a_test_feed_dict)
        summary_writer.add_summary(results["r_test_a_summary"], global_step=global_step_counter)

        b_test_fetches = {"r_loss_test": r_loss_test,
                          "r_test_b_summary": r_test_b_summary}
        b_test_feed_dict = {test_images: b_test_images,
                            test_labels: b_test_labels}
        results = sess.run(b_test_fetches, b_test_feed_dict)
        summary_writer.add_summary(results["r_test_b_summary"], global_step=global_step_counter)
    step_timer = time.time() - step_timer
    print("Step {}/{}, time: {:.5}s".format(step + 1, TOTAL_STEPS, step_timer))
    if (step + 1) % SAVE_PERIOD == 0 or (step + 1) == TOTAL_STEPS:
        print("Saving model")
        saver.save(sess, LOG_DIR + "model.ckpt")
        tf.train.export_meta_graph(LOG_DIR + "r_model.ckpt.meta",
                                   export_scope="regressor",
                                   collection_list=["pred", "image"])
        # r_saver.save(sess, LOG_DIR + "r_model.ckpt")

coord.request_stop()
coord.join(threads)
sess.close()

# train_textlog.close()
# test_textlog.close()
