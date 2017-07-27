import tensorflow as tf
import tensorflow.contrib.slim as slim


def lrelu(x, leak=0.2):
    with tf.name_scope("lrelu"):
        y = tf.maximum(x, leak * x)
        return y


def batchnorm(x, eps=1e-5, scope_ext=""):
    with tf.variable_scope("batchnorm" + scope_ext):
        channels = x.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=False)
        y = tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon=eps)
        return y


def instancenorm(x, eps=1e-5, scope_ext=""):
    with tf.variable_scope("instancenorm" + scope_ext):
        channels = x.get_shape()[3]
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.constant_initializer(0))
        mean, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        y = scale * (x - mean) / tf.sqrt(var + eps) + offset
        return y


def conv2d(x, out_channels, filter_size=3, stride=2, padding="SAME", scope_ext=""):
    with tf.variable_scope("conv2d" + scope_ext):
        return slim.conv2d(x, out_channels, kernel_size=filter_size, stride=stride, padding=padding, activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(0, 0.02),
                           biases_initializer=None)


def deconv2d(x, out_channels, filter_size=3, stride=2, padding="SAME", scope_ext=""):
    with tf.variable_scope("deconv2d" + scope_ext):
        return slim.conv2d_transpose(x, out_channels, kernel_size=filter_size, stride=stride, padding=padding,
                                     activation_fn=None, weights_initializer=tf.truncated_normal_initializer(0, 0.02),
                                     biases_initializer=None)


def residual_block(x, norm=instancenorm, scope_ext=""):
    with tf.variable_scope("residual_block" + scope_ext):
        channels = x.get_shape()[3]
        y = conv2d(x, out_channels=channels, filter_size=3, stride=1, scope_ext="_1")
        y = tf.nn.relu(norm(y, scope_ext="_1"))
        y = conv2d(y, out_channels=channels, filter_size=3, stride=1, scope_ext="_2")
        y = norm(y, scope_ext="_2")
        return y + x


def feature_extractor(x, nf=16):
    with tf.variable_scope("feature_extractor"):
        y = tf.nn.relu(conv2d(x, out_channels=nf, filter_size=3, stride=2, scope_ext="_1"))
        y = tf.nn.relu(conv2d(y, out_channels=nf * 2, filter_size=3, stride=2, scope_ext="_2"))
        y = tf.nn.relu(conv2d(y, out_channels=nf * 4, filter_size=3, stride=2, scope_ext="_3"))
        y = tf.nn.relu(conv2d(y, out_channels=nf * 8, filter_size=3, stride=2, scope_ext="_4"))
        y = tf.nn.relu(conv2d(y, out_channels=nf * 8, filter_size=3, stride=2, scope_ext="_5"))
        y = tf.contrib.layers.flatten(y)
        y = tf.layers.dense(y, units=256, activation=tf.nn.relu)
        return y


def regressor_head(x):
    with tf.variable_scope("regressor_head"):
        y = tf.layers.dense(x, units=6, activation=None)
        return y


def regressor_head_3dof(x):
    with tf.variable_scope("regressor_head"):
        y = tf.layers.dense(x, units=3, activation=None)
        return y


def generator(x, nf=16, norm=instancenorm):
    with tf.variable_scope("generator"):
        c0 = tf.pad(x, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]], mode="REFLECT")
        c1 = tf.nn.relu(norm(conv2d(c0, out_channels=nf, filter_size=7, stride=1, padding="VALID", scope_ext="_1"), scope_ext="_c1"))
        c2 = tf.nn.relu(norm(conv2d(c1, out_channels=nf * 2, filter_size=3, stride=2, scope_ext="_2"), scope_ext="_c2"))
        c3 = tf.nn.relu(norm(conv2d(c2, out_channels=nf * 4, filter_size=3, stride=2, scope_ext="_3"), scope_ext="_c3"))

        r1 = residual_block(c3, scope_ext="_1")
        r2 = residual_block(r1, scope_ext="_2")
        r3 = residual_block(r2, scope_ext="_3")
        r4 = residual_block(r3, scope_ext="_4")
        r5 = residual_block(r4, scope_ext="_5")
        r6 = residual_block(r5, scope_ext="_6")
        # r7 = residual_block(r6, scope_ext="_7")
        # r8 = residual_block(r7, scope_ext="_8")
        # r9 = residual_block(r8, scope_ext="_9")

        d1 = tf.nn.relu(norm(deconv2d(r6, out_channels=nf * 2, filter_size=3, stride=2, scope_ext="_1"), scope_ext="_d1"))
        d2 = tf.nn.relu(norm(deconv2d(d1, out_channels=nf, filter_size=3, stride=2, scope_ext="_2"), scope_ext="_d2"))
        d2 = tf.pad(d2, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]], mode="REFLECT")
        pred = conv2d(d2, out_channels=3, filter_size=7, stride=1, padding="VALID", scope_ext="_pred")
        pred = tf.nn.tanh(pred)
        return pred


def discriminator(x, nf=16, norm=instancenorm):
    with tf.variable_scope("discriminator"):
        y = lrelu(conv2d(x, out_channels=nf, filter_size=4, stride=2, scope_ext="_1"))
        y = lrelu(norm(conv2d(y, out_channels=nf * 2, filter_size=4, stride=2, scope_ext="_2"), scope_ext="_2"))
        y = lrelu(norm(conv2d(y, out_channels=nf * 4, filter_size=4, stride=2, scope_ext="_3"), scope_ext="_3"))
        y = lrelu(norm(conv2d(y, out_channels=nf * 8, filter_size=4, stride=1, scope_ext="_4"), scope_ext="_4"))
        # Why is there no activation function for the last layer? The predictions will range outside of [0,1]
        y = conv2d(y, out_channels=1, filter_size=4, stride=1, scope_ext="_pred")
        return y
