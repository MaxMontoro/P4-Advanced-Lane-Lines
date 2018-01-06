import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt


CALIBRATION_IMAGES = glob.glob('camera_cal/*.jpg') # calibration images are stored in this dir

NX = 9 # number of chessboard corners in the X axis
NY = 6 # number of chessboard corners in the Y axis

def grayscale_img(img):
    ''' Return a grayscaled version of the image '''
    new_img = img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def read_image(fname):
    ''' Reads image from a path. The returned image is in RGB colorspace. '''
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def find_chessboard_corners(img, nx=NX, ny=NY, flags=None):
    ''' Finds cheessboard corners for camera calibration '''
    ret, corners = cv2.findChessboardCorners(img, (nx, ny), flags)
    return ret, corners

def calibration_loop(images=CALIBRATION_IMAGES, show_images=False):
    ''' Main function used for calibrating the camera on a set of images.
        Optionally shows the images for visual guidance. '''
    objpoints = []
    imgpoints = []

    objp = np.zeros((NY*NX, 3), np.float32)
    objp[:,:2] = np.mgrid[0:NX,0:NY].T.reshape(-1,2)

    for i, fname in enumerate(images):
        img = read_image(fname)
        gray = grayscale_img(img)

        ret, corners = find_chessboard_corners(gray, NX, NY, None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if show_images:
                cv2.drawChessboardCorners(img, (NX,NY), corners,ret)
                cv2.imshow('img',img)
                cv2.imwrite(f'output_images/calibration_{i}.jpg', img)
                cv2.waitKey(1500)

    cv2.destroyAllWindows()
    return objpoints, imgpoints


def calibrate_camera(object_points, image_points, grayscale_img):
    ''' Given a set of object- and image points, returns a camera matrix and distrotion coefficient,
        along with the rotational vector and the transformation vectors '''
    ret, camera_mtx, dist_coeff, rot_vecs, trans_vecs = cv2.calibrateCamera(objpoints, imgpoints, grayscale_img.shape[::-1], None, None)
    return ret, camera_mtx, dist_coeff, rot_vecs, trans_vecs

def save_calibration_results(camera_mtx, dist_coeff):
    ''' Saves calibration results in a serialized object
        to a pickel file '''
    dist_pickle = {}
    dist_pickle["mtx"] = camera_mtx
    dist_pickle["dist"] = dist_coeff
    pickle.dump(dist_pickle, open("calibration.p", "wb" ))

def load_calibration_results(filename='calibration.p'):
    ''' Loads calibration data saved at an earlier session
        (so that calibration doesn't need to be re-run) '''
    with open(filename, 'rb') as calibration_file:
        dist_pickle = pickle.load(calibration_file)

    camera_mtx = dist_pickle["mtx"]
    dist_coeff = dist_pickle["dist"]
    return camera_mtx, dist_coeff

camera_mtx, dist_coeff = load_calibration_results()

def undistort_image(img, camera_mtx=camera_mtx,
                    dist_coeff=dist_coeff, newcameramtx=None):
    ''' Returns an undistorted version of the input image '''
    if newcameramtx is None:
        newcameramtx = camera_mtx
    destination_img = cv2.undistort(img, camera_mtx, dist_coeff, None, camera_mtx)
    return destination_img

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


def get_curverads_and_distance(binary_warped, leftx, rightx, ploty):
    y_eval = np.max(ploty)

    ym_per_pix = 25/720 # meters per pixel in y dimension
    xm_per_pix = 3.8/700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = int(((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0]))
    right_curverad = int(((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0]))

    # Now our radius of curvature is in meters
    car_position = 640
    left_fitx = left_fit_cr[0]*ploty**2 + left_fit_cr[1]*ploty + left_fit_cr[2]
    right_fitx = right_fit_cr[0]*ploty**2 + right_fit_cr[1]*ploty + right_fit_cr[2]

    middle = (leftx[-1] + rightx[-1])//2

    center_dist = (car_position - middle) * xm_per_pix
    center_dist = str(center_dist)[:5]
    return f'{left_curverad}m', f'{right_curverad}m', f'{center_dist}m'


def draw_text_on_image(img, sentences):
    font = cv2.FONT_HERSHEY_DUPLEX
    y_position = 40
    for text in sentences:
        cv2.putText(img, text,(40, y_position), font, 1, (255,255,255), 2, cv2.LINE_AA)
        y_position += 40
    return img

IMAGES = glob.glob('output_images/*')

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

Minv = cv2.getPerspectiveTransform(dst, src)


def find_fits(binary_warped, undistorted_image, out_img=None, margin=80, minpix=50):
    ''' Finds a polynomial that fits to the left and right lane lines.
        Uses a sliding window approach '''
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base, rightx_base = np.argmax(histogram[:midpoint]), np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = np.int(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

    leftx_current, rightx_current = leftx_base, rightx_base

    left_lane_inds, right_lane_inds = [], []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        if out_img is not None:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

def plot_polyfit_on_warped_image(binary_warped, left_fit, right_fit, out_img, show_image=False):
    ''' Plots the polynomial on the warped image '''
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if show_image:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 70
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
    left_fit[1]*nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    left_fit, right_fit = np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    if show_image:
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
    return result, left_fitx, right_fitx, ploty


def draw_poly(binary_warped, undistorted_image, left_fitx, right_fitx, ploty):
    ''' Draws the polyomial on the undistorted image '''
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts_left]), (0, 0, 255))
    cv2.fillPoly(color_warp, np.int_([pts_right]), (0, 0, 255))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,240, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (binary_warped.shape[1], binary_warped.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_image, 1, newwarp, 0.3, 0)

    l_curv_radius, r_curve_radius, distance = get_curverads_and_distance(binary_warped, left_fitx, right_fitx, ploty)
    result = draw_text_on_image(result, [l_curv_radius, r_curve_radius, distance])

    return result


def sliding_window(binary_warped, undistorted_image, out_img=None):
    ''' Main function for the sliding window search and polynomial fitting '''
    left_fit, right_fit = find_fits(binary_warped, undistorted_image, out_img=out_img)
    result, left_fitx, right_fitx, ploty = plot_polyfit_on_warped_image(
            binary_warped, left_fit, right_fit, out_img=undistorted_image, show_image=False
        )
    return draw_poly(binary_warped, undistorted_image, left_fitx, right_fitx, ploty)

from moviepy.editor import VideoFileClip

CWD = os.getcwd()

def pipeline(image, show_images=False):
    ''' Pipeline for processing frames from the video '''
    undist = undistort_image(image)
    thresholded = apply_mask(apply_thresholds(undist))
    unwarped, M, Minv = unwarp_image(thresholded, src, dst)
    result = sliding_window(unwarped, undist)
    if show_images:
        plt.imshow(image)
        plt.show()
        plt.imsave('output_images/original.jpg', image)
        plt.imshow(undist)
        plt.show()
        plt.imsave('output_images/undistorted.jpg', undist)
        plt.imshow(apply_mask(undist))
        plt.show()
        plt.imsave('output_images/masked.jpg', apply_mask(undist))
        plt.imshow(thresholded, cmap='gray')
        plt.show()
        plt.imsave('output_images/thresholded.jpg', thresholded, cmap='gray')
        plt.imshow(unwarped, cmap='gray')
        plt.show()
        plt.imsave('output_images/unwarped.jpg', unwarped, cmap='gray')
        plt.imshow(result)
        plt.show()
        plt.imsave('output_images/result.jpg', result)
        cv2.imshow('img',result)
        cv2.waitKey(1)
    return result


def process_image(image):
    result = pipeline(image, show_images=False)
    return result

def process_video(path):
    filename = os.path.basename(path)
    output = os.path.join(CWD, 'test_videos_output/6_{}'.format(filename))
    clip = VideoFileClip(path)
    processed = clip.fl_image(process_image)
    processed.write_videofile(output, audio=False)


if __name__ == '__main__':
    process_video('project_video.mp4')
