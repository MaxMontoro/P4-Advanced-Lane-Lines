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


if __name__ == '__main__':
    ''' If the script is invoked directly, it calibrates the camera and shows images for demonstration '''

    img = read_image('camera_cal/calibration3.jpg')
    gray = grayscale_img(img)

    objpoints, imgpoints = calibration_loop(CALIBRATION_IMAGES, show_images=False)
    ret, camera_mtx, dist_coeff, rot_vecs, trans_vecs = calibrate_camera(objpoints, imgpoints, gray)

    h,  w = img.shape[:2]

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_mtx, dist_coeff, (w,h), 0, (w,h))
    undistorted_image = undistort_image(img, camera_mtx, dist_coeff,newcameramtx=newcameramtx)

    plt.imsave('output_images/calibration_undist3.jpg', undistorted_image)
    #cv2.imshow('img',undistorted_image)
    #cv2.waitKey(25000)
