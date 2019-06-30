#!/usr/bin/python
import cv2
import numpy as np


def get_gradient(image):
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


if __name__ == '__main__':

    # Read 16-bit color image.
    # This is an image in which the two channels are
    # concatenated horizontally.
    # The first frame of a .pcoraw video of a adult rat heart.
    im = cv2.imread("images/09-250_0001", cv2.IMREAD_GRAYSCALE)

    # Find the width and height of the color image
    im_size = im.shape
    print(im_size)
    height = im_size[0]
    width = int(im_size[1] / 2)

    # Extract the two channels from the gray scale image
    # and merge the two channels into one color image
    im_color = np.zeros((height, width, 2), dtype=np.uint16)
    for i in range(0, 2):
        im_color[:, :, i] = im[i * height:(i + 1) * height, :]

    # Allocate space for aligned image
    im_aligned = np.zeros((height, width, 3), dtype=np.uint16)

    # The right channel will be aligned to the left channel.
    # So copy the left channel
    im_aligned[:, :, 0] = im_color[:, :, 0]

    # Define motion model
    warp_mode = cv2.MOTION_TRANSLATION
    # Set the warp matrix to identity.
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Warp the right channel to the left channel
    print('Hi')
    (cc, warp_matrix) = cv2.findTransformECC(get_gradient(im_color[:, :, 0]), get_gradient(im_color[:, :, 1]),
                                             warp_matrix, warp_mode, criteria, None, 5)
    # Use Affine warp when the transformation is not a Homography
    im_aligned[:, :, 1] = cv2.warpAffine(im_color[:, :, 1], warp_matrix, (width, height),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    print(warp_matrix)

    # Show final output
    cv2.imshow("Color Image", im_color)
    cv2.imshow("Aligned Image", im_aligned)
    cv2.waitKey(0)
