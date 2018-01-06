import os

from camera_calibration import undistort_image
from perspective import unwarp_image, src, dst
from color_thresholds import apply_thresholds, apply_mask
from sliding_window import *
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
