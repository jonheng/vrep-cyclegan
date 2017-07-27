import tensorflow as tf
import time

r_only = "./log/3dof_regressor_combined/model.ckpt"
r_model = "./log/a2b2a_3dof/r_model.ckpt"
full_model = "./log/a2b2a_3dof/model.ckpt"


def time_model_load(load_model_path):
    timer = time.time()
    sess = tf.Session()
    saver = tf.train.import_meta_graph(load_model_path + ".meta")
    saver.restore(sess, load_model_path)
    image_op = tf.get_collection("image")[0]
    pred_op = tf.get_collection("pred")[0]
    timer = time.time() - timer
    print("Model path: " + load_model_path)
    print("Time taken: {:.5}s".format(timer))

time_model_load(r_only)
time_model_load(r_model)
time_model_load(full_model)
