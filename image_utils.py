import numpy as np
import matplotlib.pyplot as plt
from skimage import feature


def h5py_to_array(dataset, img_shape):
    # img_shape as a tuple
    # assumes input is in the uint8 format
    # converts into a standard float32 in range[-1,1]
    images = np.array(dataset)
    images = (np.float32(images) - 127.0) / 128.0
    images = np.reshape(images, (images.shape[0],) + img_shape)
    return images


def h5py_to_array2(dataset, img_shape):
    # converts into images to a range of [0,1]
    images = np.array(dataset)
    images = np.float32(images) / 255.0
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


def display(img):
    # origin lower flips the image from top to bottom (mirror around x-axis)
    img = (img + 1.0) / 2.0
    return plt.imshow(img, origin='lower')


def display_2images(image1, image2):
    fig = plt.figure()
    a = fig.add_subplot(1,2,1)
    a = display(image1)
    b = fig.add_subplot(1,2,2)
    b = display(image2)
    return