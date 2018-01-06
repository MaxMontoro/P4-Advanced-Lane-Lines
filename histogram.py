import glob

import numpy as np
import matplotlib.pyplot as plt

from camera_calibration import read_image

IMAGES = glob.glob('output_images/*')

if __name__ == '__main__':
    for image in IMAGES:
        img = read_image(image)
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        plt.plot(histogram)
        plt.show()
