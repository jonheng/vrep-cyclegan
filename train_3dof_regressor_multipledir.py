# Trains a neural network to perform regression on joint velocity labels for a 3dof arm
# Testing how performance and losses change with size of dataset available

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from model import *
from PIL import Image

""" HYPER-PARAMETERS """
NUM_EPISODES = 15707 + 16384
TOTAL_IMAGES = NUM_EPISODES * 16
BATCH_SIZE = 16
NUM_EPOCHS = 3
MAX_TRAINING_IMAGES = TOTAL_IMAGES * NUM_EPOCHS
TOTAL_STEPS = MAX_TRAINING_IMAGES / BATCH_SIZE
RESTORE = False
NUM_THREADS = 2
IMG_SHAPE = [128, 128, 3]
NUM_JOINTS = 3

TRAIN_DIR = ["./datasets/3dof-arm-grid/", "./datasets/3dof-arm/"]
TEST_DIR = "./datasets/3dof-arm-test/"
LOG_DIR = "./log/3dof_regressor_combined/"

TEST_PERIOD = 250
SAVE_PERIOD = 10000

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

train_images_dir = []
train_labels_path = []
train_images_paths = []
for dir in TRAIN_DIR:
    train_images_dir.append(dir + "images/")
    train_labels_path.append(dir + "joint_vel.txt")
for dir in train_images_dir:
    train_images_paths += [dir + "image" + str(i) + ".jpg" for i in range(1, len(os.listdir(dir)) + 1)]
train_labels_array = np.empty(shape=(0, 3), dtype=np.float32)
for label_path in train_labels_path:
    train_labels_array = np.concatenate([train_labels_array, read_labels(label_path)])
print len(train_images_paths)
print train_labels_array.shape
train_images = tf.convert_to_tensor(train_images_paths, dtype=tf.string)
train_labels = tf.convert_to_tensor(train_labels_array, dtype=tf.float32)
train_input_queue = tf.train.slice_input_producer([train_images, train_labels],
                                                  num_epochs=NUM_EPOCHS,
                                                  shuffle=True)
train_image = tf.image.decode_jpeg(tf.read_file(train_input_queue[0]))
train_image = preprocess_image(train_image)
train_image.set_shape(IMG_SHAPE)
train_label = train_input_queue[1]
train_image_batch, train_label_batch = tf.train.batch([train_image, train_label],
                                                      batch_size=BATCH_SIZE,
                                                      allow_smaller_final_batch=True,
                                                      num_threads=NUM_THREADS)


test_images_dir = TEST_DIR + "images/"
test_labels_path = TEST_DIR + "joint_vel.txt"
test_size = len(os.listdir(test_images_dir))
print("Test size: {}".format(test_size))
test_images_paths = [test_images_dir + "image" + str(i) + ".jpg" for i in range(1, test_size + 1)]
test_images_array = read_images(test_images_paths)
test_labels_array = read_labels(test_labels_path, test_size)

test_images = tf.placeholder(dtype=tf.float32, shape=[None, ] + IMG_SHAPE)
test_labels = tf.placeholder(dtype=tf.float32, shape=[None, NUM_JOINTS])


""" model """
with tf.variable_scope("regressor") as scope:
    feature_layer = feature_extractor(train_image_batch)
    pred = regressor_head_3dof(feature_layer)
    scope.reuse_variables()
    test_feature_layer = feature_extractor(test_images)
    test_pred = regressor_head_3dof(test_feature_layer)

global_vars = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(0.001)

""" step counters """
global_step = tf.contrib.framework.get_or_create_global_step()
incr_global_step = tf.assign(global_step, global_step + 1)

""" loss """
mse = tf.reduce_mean(tf.square(train_label_batch - pred))
test_mse = tf.reduce_mean(tf.square(test_labels - test_pred))
grads = optimizer.compute_gradients(mse, var_list=global_vars)
# Increment global step every time a train is done
with tf.control_dependencies([incr_global_step]):
    train_op = optimizer.apply_gradients(grads, global_step=global_step)

""" summaries """
loss_summaries = tf.summary.scalar("train_mse_loss_epoch", mse)
test_summaries = tf.summary.scalar("test_mse_loss_epoch", test_mse)

""" collection """
tf.add_to_collection("pred", test_pred)
tf.add_to_collection("image", test_images)

""" textfile logs """
train_textlog = open(LOG_DIR + "train_loss.txt", "w")
test_textlog = open(LOG_DIR + "test_loss.txt", "w")

""" execution """
sess = tf.Session()

""" save / restore """
saver = tf.train.Saver()
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
    _, train_loss, train_summary_str = sess.run([train_op, mse, loss_summaries])
    print("Step {}/{}, train loss: {}".format(step + 1, TOTAL_STEPS, train_loss))
    summary_writer.add_summary(train_summary_str, step)
    train_textlog.write(str(step + 1) + " " + str(round(train_loss, 6)) + "\n")

    if (step + 1) % TEST_PERIOD == 0 or (step + 1) == TOTAL_STEPS:
        test_loss, test_summary_str = sess.run([test_mse, test_summaries],
                                               feed_dict={test_images: test_images_array,
                                                          test_labels: test_labels_array})
        print("Test loss: ", test_loss)
        summary_writer.add_summary(test_summary_str, step)
        test_textlog.write(str(step + 1) + " " + str(round(test_loss, 6)) + "\n")
    if (step + 1) % SAVE_PERIOD == 0 or (step + 1) == TOTAL_STEPS:
        print("Saving model")
        saver.save(sess, LOG_DIR + "model.ckpt")


coord.request_stop()
coord.join(threads)
sess.close()

train_textlog.close()
test_textlog.close()
