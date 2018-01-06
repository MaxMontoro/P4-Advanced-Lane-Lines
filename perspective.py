import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

from camera_calibration import read_image, undistort_image

IMAGES = glob.glob('test_images/test1.jpg')

image = read_image('test_images/straight_lines1.jpg')
undistorted = undistort_image(image)

h, w = undistorted.shape[:2]

# define source and destination points for transform
'''
src = np.float32([(574,473),
                  (753,472),
                  (368,629),
                  (1039,653)])
dst = np.float32([(450,0),
                  (w-240,0),
                  (450,h-55),
                  (w-240,h-25)])
'''

src = np.float32([(602,458),
                  (731,456),
                  (368,629),
                  (1039,653)])
dst = np.float32([(450,0),
                  (w-240,0),
                  (450,h-55),
                  (w-240,h-25)])


def unwarp_image(img, src, dst):
    ''' Unwarps (perspective transforms) the input image,
        matching source points to destination points '''
    h,w = img.shape[:2]
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv


if __name__ == '__main__':
    for image in IMAGES:
        image = read_image(image)
        undistorted = undistort_image(image)
        image, M, Minv = unwarp_image(undistorted, src, dst)
        plt.imshow(image, cmap='gray')
        plt.show()
