import cv2
import numpy as np
from scipy.stats import wasserstein_distance


def earth_movers_distance(im1, im2):
    hist1 = get_histogram(im1)
    hist2 = get_histogram(im2)

    return wasserstein_distance(hist1, hist2)

def get_histogram(img):
    '''
    Get the histogram of an image. For an 8-bit, grayscale image, the
    histogram will be a 256 unit vector in which the nth value indicates
    the percent of the pixels in the image with the given darkness level.
    The histogram's values sum to 1.
    '''
    h, w = img.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
    return np.array(hist) / (h * w)
