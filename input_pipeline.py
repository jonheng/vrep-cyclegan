from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def read_labels(label_path):
    """
    Input: Label file path
    Returns: List of labels 
    """
    file = open(label_path, "r")
    data = file.readlines()
    size = len(data)
    output = np.empty(shape=(size, 3), dtype=np.float32)
    for i in range(size):
        line_array = np.array(data[i].rstrip().split(" "), dtype=np.float32)
        output[i, :] = line_array[1:4]

    return output

# image_paths = glob("./datasets/3dof-arm-test/images/*.jpg")
image_paths = ["./datasets/3dof-arm-test/images/image" + str(i) + ".jpg" for i in range(1, 321)]
image_tensor = tf.convert_to_tensor(image_paths, dtype=tf.string)

label_path = "./datasets/3dof-arm-test/joint_vel.txt"
label_array = read_labels(label_path)
label_tensor = tf.convert_to_tensor(label_array, dtype=tf.float32)


input_queue = tf.train.slice_input_producer([image_tensor, label_tensor],
                                            shuffle=False)

image = tf.read_file(input_queue[0])
image = tf.image.decode_jpeg(image, channels=3)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    image_eval = sess.run(image)
    plt.imshow(image_eval)
    plt.show()

    coord.request_stop()
    coord.join(threads)
    sess.close()
