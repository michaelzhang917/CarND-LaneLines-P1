import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import figure, axes, pie, title, show
import numpy as np
import cv2
import time
import math


def color_selection(image, boundaries, type):
    # loop over the boundaries
    mask = np.zeros_like(image[:,:,1])
    if (type == "HSV"):
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    else:
        image_copy = image.copy()
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.bitwise_or(mask, cv2.inRange(image_copy, lower, upper))
    return cv2.bitwise_and(image, image, mask = mask)


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, yrange, color=[255, 0, 0], thickness=4):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left_y = 0
    left_x = 0
    right_y = 0
    right_x = 0
    left_count = 0
    right_count = 0
    left_angle = 0
    right_angle = 0
    global left_info
    global right_info


    for line in lines:
        for x1, y1, x2, y2 in line:
            if (x2 == x1) :
                theta = np.pi/2
            else:
                theta = np.arctan((y2 - y1) / (x2 - x1))
            if (np.pi / 6 < abs(theta) < np.pi / 4):
                (x1, y1), _ = extend_line((x1, y1), theta, yrange)
                if (theta < 0):
                    if ((not left_info) or abs(theta - left_info[2]) < np.pi/24):
                        left_y = left_y + y1 + y2
                        left_x = left_x + x1 + x2
                        left_count = left_count + 2
                        left_angle = left_angle + theta
                else:
                    if ((not right_info) or abs(theta - right_info[2]) < np.pi/24):
                        right_y = right_y + y1 + y2
                        right_x = right_x + x1 + x2
                        right_count = right_count + 2
                        right_angle = right_angle + theta

    if (left_count > 0):
        left_y = left_y / left_count
        left_x = left_x / left_count
        left_angle = left_angle / left_count * 2
        ((xhat1, yhat1), (xhat2, yhat2)) = extend_line((left_x, left_y), left_angle, yrange)
        cv2.line(img, (xhat1, yhat1), (xhat2, yhat2), color, thickness)
    else :
        left_x = left_info[0]
        left_y = left_info[1]
        left_angle = left_info[2]
        ((xhat1, yhat1), (xhat2, yhat2)) = extend_line((left_x, left_y), left_angle, yrange)
        cv2.line(img, (xhat1, yhat1), (xhat2, yhat2), color, thickness)
    left_info = (left_x, left_y, left_angle)

    if (right_count > 0):
        right_y = right_y / right_count
        right_x        = right_x / right_count
        right_angle = right_angle / right_count * 2
        ((xhat1, yhat1), (xhat2, yhat2)) = extend_line((right_x, right_y), right_angle, yrange)
        cv2.line(img, (xhat1, yhat1), (xhat2, yhat2), color, thickness)
    else :
        right_x = right_info[0]
        right_y = right_info[1]
        right_angle = right_info[2]
        ((xhat1, yhat1), (xhat2, yhat2)) = extend_line((right_x, right_y), right_angle, yrange)
        cv2.line(img, (xhat1, yhat1), (xhat2, yhat2), color, thickness)
    right_info = (right_x, right_y, right_angle)




def extend_line(pt, angle, yrange):
    x = pt[0]
    y = pt[1]
    k = 1 / np.tan(angle)
    yhat1 = yrange[0]
    xhat1 = int((yhat1 - y) * k + x)
    yhat2 = yrange[1]
    xhat2 = int((yhat2 - y) * k + x)
    return ((xhat1, yhat1), (xhat2, yhat2))



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, yrange):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines, yrange)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


import os

files = os.listdir("test_images/")
low_threshold = 50
high_threshold = 150
kernel_size = 5
rho = 2
theta = np.pi / 180
threshold = 15
min_line_length = 50
max_line_gap = 50

left_info = []
right_info = []
color_boundaries_RGB = [ ([220, 220, 220], [255, 255, 255]), ([180, 160, 0], [255, 255, 255])]
color_boundaries_HSV = [ ([0, 0, 200], [180, 255, 255]), ([20, 100, 100], [30, 255, 255])]
cap = cv2.VideoCapture("challenge.mp4")


#for file in files:
    #image = mpimg.imread("test_images/" + file)
    #imshape = image.shape

for fn in range(0, 252):
    print(fn)
    cap.set(1,fn);
    ret, frame = cap.read();
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imshape = image.shape
    output =color_selection(image, color_boundaries_HSV, "HSV")
    #cv2.imshow('img',  cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    #cv2.waitKey()
    output = color_selection(output, color_boundaries_RGB, "")
    #cv2.imshow('img', cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    #cv2.waitKey()
    #fig = plt.figure()
    #plt.imshow(image)
    #fig.savefig("results/" + file + '_original.png')
    #cv2.imshow('img',  cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #cv2.waitKey()

    #cv2.imshow('img', cv2.cvtColor (output, cv2.COLOR_BGR2RGB))
    #cv2.waitKey()

    gray = grayscale(output)
    #cv2.imshow('img', gray)
    #cv2.waitKey()
    #plt.imshow(gray, cmap='gray')
    #fig.savefig("results/" + file + '_gray.png')

    blur_gray = gaussian_blur(gray, kernel_size)
    edges = canny(blur_gray, low_threshold, high_threshold)
    #cv2.imshow('img', edges )
    #cv2.waitKey()
    #plt.imshow(edges, cmap='Greys_r')
    #fig.savefig("results/" + file  + '_edge.png')
    Ymin = 450
    vertices = np.array([[(0, imshape[0]), (450, 320), (490, 320), (imshape[1], imshape[0])]], dtype=np.int32)
   # vertices = np.array([[(0, imshape[0]), (450, 320), (460, 320), (460, imshape[0])]], dtype=np.int32)
    yrange = (Ymin, imshape[0])
    region_select = region_of_interest(edges, vertices)
    line_image = hough_lines(region_select, rho, theta, threshold, min_line_length, max_line_gap, yrange)
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the edge image
    combo = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    # combo = process_image(image)

    #plt.imshow(combo)
    #fig.savefig("results/" + file + '_combo.png')
    cv2.imshow('img', cv2.cvtColor(combo, cv2.COLOR_BGR2RGB))
    cv2.waitKey()

