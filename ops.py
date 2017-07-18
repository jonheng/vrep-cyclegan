import tensorflow as tf


def l2_loss(a, b, weight=1.0):
    return tf.reduce_mean((a - b) ** 2) * weight


def l1_loss(a, b, weight=1.0):
    return tf.reduce_mean(tf.abs(a - b)) * weight
