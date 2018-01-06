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
# source points edited for project resubmission
src = np.float32([(573,460),
                  (705,462),
                  (310,682),
                  (1090,682)])
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])

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
