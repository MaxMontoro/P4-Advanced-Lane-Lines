import glob
import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from camera_calibration import read_image, grayscale_img

THRESHOLDS = ['r_select', 'hls_select', 'dir_threshold', 'mag_thresh', 'abs_sobel_thresh']
IMAGES = glob.glob('test_images/test1.jpg')
IMAGES.append('test_images/straight_lines1.jpg')
IMAGES.append('test_images/shady.png')

ksize = 7


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    ''' Applies Sobel threshold along the X or Y direction.
        Returns binary image '''
    img = img.copy()
    gray = grayscale_img(img)    # Convert to grayscale
    # Apply x or y gradient with the OpenCV Sobel() function and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=9, mag_thresh=(40, 255)):
    ''' Returns the magnitude of the gradient
        for a given sobel kernel size and threshold values '''
    img = img.copy()
    # Convert to grayscale
    gray = grayscale_img(img)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=9, thresh=(0, np.pi/2)):
    ''' Applies a direction thresholded Sobel operator '''
    img = img.copy()
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_select(img, thresh=(40, 255)):
    ''' Selects the S channell from HLS colorspace
        and applies a threshold on it '''
    img = img.copy()
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def r_select(img, thresh=(40, 255)):
    ''' Selects the R channel from the RGB colorspace
        and applies a threshold on it '''
    R = img[:,:,0]
    binary = np.zeros_like(R)
    binary[(R > thresh[0]) & (R <= thresh[1])] = 1
    return binary

def g_select(img, thresh=(40, 255)):
    ''' Selects the G channel from the RGB colorspace
        and applies a threshold on it '''
    G = img[:,:,1]
    binary = np.zeros_like(G)
    binary[(G > thresh[0]) & (G <= thresh[1])] = 1
    return binary

def b_select(img, thresh=(40, 255)):
    ''' Selects the B channel from the RGB colorspace
        and applies a threshold on it '''
    B = img[:,:,1]
    binary = np.zeros_like(B)
    binary[(B > thresh[0]) & (B <= thresh[1])] = 1
    return binary

def rgb_ratio_select(img, ratio=(1, .8, .3), approx_value=None):
    ''' Select pixels that have the RGB ratio and approximate value
        specified with the parameters '''
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    binary = np.zeros_like(B)
    binary[ ( np.around(G/R, 1) == ratio[1] ) & ( np.around(B/R, 1) == ratio[2] )] = 1
    if approx_value:
        binary[ ( np.around(G/255,1) != approx_value) &
            ( np.around(B/255,1) != approx_value)] = 0
    return binary



def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending
    # on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill
    # color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def mask_to_use(image):
    return np.array([[image.shape[1] // 2 - 130, int(image.shape[0] * .58)],
                     [0, image.shape[0]],
                     [image.shape[1], image.shape[0]],
                     [image.shape[1] // 2 + 30, int(image.shape[0] * .58)]])

THRESHOLD = {
             'mag': (0, 255),
             'dir': (np.pi/5, .4*np.pi),
             'gradx': (0, 255),
             'grady': (0, 255),
             'r': (220, 255),
             'g': (180, 255),
             'b': (60, 255),
             's': (10, 255)
            }


def apply_thresholds(img, thresholds=THRESHOLDS):
    ''' Defines and applies thresholds to the image.
        Returns a binary image where pixels which meet the threshold values
        are white, others are black. '''
    gradx = abs_sobel_thresh(img, orient='x', thresh_min=THRESHOLD['gradx'][0],
                                              thresh_max=THRESHOLD['gradx'][1])
    grady = abs_sobel_thresh(img, orient='y', thresh_min=THRESHOLD['grady'][0],
                                              thresh_max=THRESHOLD['grady'][1])

    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=THRESHOLD['mag'])
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=THRESHOLD['dir'])
    r_binary = r_select(img, thresh=THRESHOLD['r'])
    g_binary = g_select(img, thresh=THRESHOLD['g'])

    # thresholds to pick up shades of yellow and white
    rgb_ratio_binary = rgb_ratio_select(img)
    rgb_ratio_binary_2 = rgb_ratio_select(img, ratio=(1,.8,.4))
    rgb_ratio_binary_3 = rgb_ratio_select(img, ratio=(1,.9,.5))
    rgb_ratio_binary_4 = rgb_ratio_select(img, ratio=(1,.9,.6))
    rgb_ratio_binary_5 = rgb_ratio_select(img, ratio=(1,.9,.7))

    yellow = ((r_binary == 1) & (g_binary == 1))
    white_binary = rgb_ratio_select(img, ratio=(1,1,1), approx_value=.7)
    white_binary2 = rgb_ratio_select(img, ratio=(1,1,1), approx_value=.8)
    white_binary3 = rgb_ratio_select(img, ratio=(1,1,1), approx_value=.9)

    # dark gray spots to be left out
    dark_gray_binary = rgb_ratio_select(img, ratio=(1,1,1), approx_value=.3)

    combined = np.zeros_like(r_binary)

    combined[ ((white_binary==1) |
               (white_binary2==1) |
               (white_binary3==1) |
               (yellow == 1) |
               (rgb_ratio_binary==1) |
               (rgb_ratio_binary_2==1)|
               (rgb_ratio_binary_3==1) |
               (rgb_ratio_binary_4 == 1)) &
               ((dark_gray_binary == 0) & (gradx == 1) & (dir_binary == 1))
                ] = 1

    return combined

def apply_mask(img):
    ''' Apply region of interest mask on the image '''
    mask = mask_to_use(img)
    masked_image = region_of_interest(img, [mask])
    return masked_image


if __name__ == '__main__':
    ''' If the script is invoked directly, applies thresholds the IMAGES array '''
    for image in ['output_images/original.jpg']:
        image = read_image(image)
        thresholded = apply_thresholds(image)
        plt.imshow(thresholded, cmap='gray')
        plt.imsave('output_images/thresholded1.jpg', thresholded, cmap='gray')
