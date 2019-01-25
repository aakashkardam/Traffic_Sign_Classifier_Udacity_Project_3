# This is a collection of image augmentation techniques. More techniques can be added to this list and used directly in the jupyter-notebook
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import numpy as np
import tensorflow as tf
import scipy

def rotate_randomly(img):
  random_degree = random.uniform(-180, 180)
  return sk.transform.rotate(img, random_degree)

def add_noise_randomly(img):
  #noise = tf.random_normal(shape=tf.shape(img), mean=0.0, stddev=1.0)
  #temp = img+noise
  #tf.reshape(temp,img.shape)
  return sk.util.random_noise(img)

def translate(image, offset, isseg=False):
    order = 0 if isseg == True else 5

    return scipy.ndimage.interpolation.shift(image, (int(offset[0]), int(offset[1]), 0), order=order, mode='nearest')


def flip_upside_down(img):
  return np.flipud(img)

def flip_left_2_right(img):
  return np.fliplr(img)
