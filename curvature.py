import numpy as np
import cv2

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
