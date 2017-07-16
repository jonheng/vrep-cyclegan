import numpy as np
from skimage import feature


def h5py_to_array(dataset, img_shape):
    # img_shape as a tuple
    # assumes input is in the uint8 format
    # converts into a standard float32 in range[-1,1]
    images = np.array(dataset)
    images = (np.float32(images) - 127.0) / 128.0
    images = np.reshape(images, (images.shape[0],) + img_shape)
    return images


def tint_images(images, filter=[1.0, 1.0, 1.0]):
    return filter*images


def canny_edges(images, sigma=1):
    n, h, w, c = images.shape
    edge_images = np.zeros_like(images)
    for i in range(n):
        edge_channels = np.zeros(shape=(h, w, c), dtype=np.float32)
        for channel in range(c):
            edge_channels[:, :, channel] = feature.canny(images[i, :, :, channel], sigma=sigma)
        edge_images[i] = edge_channels
    return edge_images