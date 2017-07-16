import tensorflow as tf


def lrelu(x, leak=0.2):
    with tf.name_scope("lrelu"):
        y = tf.maximum(x, leak * x)
        return y


def batchnorm(x, variance_epsilon=1e-5, scope_ext=""):
    with tf.variable_scope("batchnorm" + scope_ext):
        channels = x.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=False)
        y = tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return y


def instancenorm(x, eps=1e-5, scope_ext=""):
    with tf.variable_scope("instancenorm" + scope_ext):
        channels = x.get_shape()[3]
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.constant_initializer(0))
        mean, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        y = scale * (x - mean) / tf.sqrt(var + eps) + offset
        return y


def conv2d(x, out_channels, pad=1, filter_size=3, stride=2, scope_ext=""):
    with tf.variable_scope("conv2d" + scope_ext):
        in_channels = x.get_shape()[3]
        filter = tf.get_variable("filter", [filter_size, filter_size, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        y = tf.pad(x, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
        y = tf.nn.conv2d(y, filter=filter, strides=[1, stride, stride, 1], padding="VALID")
        return y


def residual_block(x, scope_ext=""):
    with tf.variable_scope("residual_block" + scope_ext):
        channels = tf.shape(x)[3]
        y = conv2d(x, out_channels=channels, pad=1, filter_size=3, stride=1, scope_ext="_1")
        y = tf.nn.relu(batchnorm(y))
        y = conv2d(y, out_channels=channels, pad=1, filter_size=3, stride=1, scope_ext="_2")
        y = batchnorm(y)
        return y + x


def feature_extractor(x, nf=16):
    with tf.variable_scope("feature_extractor"):
        y = lrelu(conv2d(x, out_channels=nf, scope_ext="_1"))
        y = lrelu(batchnorm(conv2d(y, out_channels=nf * 2, scope_ext="_2"), scope_ext="_2"))
        y = lrelu(batchnorm(conv2d(y, out_channels=nf * 4, scope_ext="_3"), scope_ext="_3"))
        y = lrelu(batchnorm(conv2d(y, out_channels=nf * 4, scope_ext="_4"), scope_ext="_4"))
        y = lrelu(batchnorm(conv2d(y, out_channels=nf * 4, scope_ext="_5"), scope_ext="_5"))
        y = tf.contrib.layers.flatten(y)
        return y


def feature_extractor2(x, nf=16):
    with tf.variable_scope("feature_extractor"):
        y = lrelu(conv2d(x, out_channels=nf, scope_ext="_1"))
        y = lrelu(conv2d(y, out_channels=nf * 2, scope_ext="_2"), scope_ext="_2")
        y = lrelu(conv2d(y, out_channels=nf * 4, scope_ext="_3"), scope_ext="_3")
        y = lrelu(conv2d(y, out_channels=nf * 8, scope_ext="_4"), scope_ext="_4")
        y = lrelu(conv2d(y, out_channels=nf * 8, scope_ext="_5"), scope_ext="_5")
        y = tf.contrib.layers.flatten(y)
        return y


def feature_extractor3(x, nf=16):
    with tf.variable_scope("feature_extractor"):
        y = tf.nn.relu(conv2d(x, out_channels=nf, scope_ext="_1"))
        y = tf.nn.relu(conv2d(y, out_channels=nf * 2, scope_ext="_2"), scope_ext="_2")
        y = tf.nn.relu(conv2d(y, out_channels=nf * 4, scope_ext="_3"), scope_ext="_3")
        y = tf.nn.relu(conv2d(y, out_channels=nf * 8, scope_ext="_4"), scope_ext="_4")
        y = tf.nn.relu(conv2d(y, out_channels=nf * 8, scope_ext="_5"), scope_ext="_5")
        y = tf.contrib.layers.flatten(y)
        y = tf.layers.dense(y, units=256, activation=tf.nn.relu)
        return y

def regressor_head(x):
    with tf.variable_scope("regressor_head"):
        y = tf.layers.dense(x, units=6, activation=None)
        return y