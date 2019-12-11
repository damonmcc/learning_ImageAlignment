#!/usr/bin/python
import cv2
import numpy as np
from tifffile import imsave
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.colors as colors
import ScientificColourMaps5 as SCMaps

# Colors and colormaps
color_vm, color_ca = ['#FF9999', '#99FF99']
# cmap_vm = SCMaps.bilbao.reversed()
# cmap_ca = SCMaps.bamako
cmap_vm = SCMaps.lajolla.reversed()
cmap_ca = SCMaps.davos


def get_gradient(image):
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


if __name__ == '__main__':

    # Read 16-bit color image.
    # This is an image in which the two channels are concatenated horizontally.
    # The first frame of a .pcoraw video of an adult rat heart,
    # RH237/Vm emission on the left and Rhod2/Ca emission on the right.
    im_filename = '11-150_0001.tif'
    im = cv2.imread(im_filename, cv2.IMREAD_GRAYSCALE)

    # Find the width and height of the color image
    im_size = im.shape
    print('Original image size ' + str(im_size))
    height = im_size[0]
    width = int(im_size[1] / 2)

    # Split the full image into a left image and right image
    im_dual = np.zeros((height, width, 2), dtype=np.uint16)
    for i in range(0, 2):
        im_dual[:, :, i] = im[:, i * width:(i + 1) * width]

    # Allocate space for aligned image
    img_dual_aligned = np.zeros((height, width, 2), dtype=np.uint16)

    # The right channel will be aligned to the left channel.
    # So copy the left channel
    img_dual_aligned[:, :, 0] = im_dual[:, :, 0]

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

    # Enhanced Correlation Coefficient (ECC)
    # Run the ECC algorithm. The results are stored in warp_matrix.
    start = time.time()
    # (cc, warp_matrix) = cv2.findTransformECC(im_color[:, :, 0], im_color[:, :, 1],
    #                                          warp_matrix, warp_mode, criteria, None, 5)
    # Warp the right channel to the left channel
    (cc, warp_matrix) = cv2.findTransformECC(get_gradient(im_dual[:, :, 0]), get_gradient(im_dual[:, :, 1]),
                                             warp_matrix, warp_mode, criteria, None, 5)
    # Use Affine warp when the transformation is not a Homography
    img_dual_aligned[:, :, 1] = cv2.warpAffine(im_dual[:, :, 1], warp_matrix, (width, height),
                                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    print('warp matrix:')
    print(warp_matrix)

    end = time.time()
    print('Alignment time (s): ', end - start)

    # Show final output
    fig_alignment = plt.figure(figsize=(12, 4))  # _ x _ inch page
    gs0 = fig_alignment.add_gridspec(1, 2)  # 1 rows, 2 column

    # Unaligned images
    gs_unaligned = gs0[0].subgridspec(2, 1)  # 2 row, 1 columns
    # Full
    ax_unaligned_full = fig_alignment.add_subplot(gs_unaligned[0])
    cmap_frame = SCMaps.grayC.reversed()
    img_unaligned_full = ax_unaligned_full.imshow(im, cmap=cmap_frame)
    # Split
    gs_unaligned_split = gs_unaligned[1].subgridspec(1, 2)  # 1 rows, 2 column
    # Vm
    img_vm = img_dual_aligned[:, :, 0]
    ax_unaligned_split_L = fig_alignment.add_subplot(gs_unaligned_split[0])
    cmap_norm_vm = colors.Normalize(vmin=img_vm.min(), vmax=img_vm.max())
    img_unaligned_split_L = ax_unaligned_split_L.imshow(img_vm, cmap=cmap_vm, norm=cmap_norm_vm)
    # Ca
    img_ca = img_dual_aligned[:, :, 1]
    ax_unaligned_split_R = fig_alignment.add_subplot(gs_unaligned_split[1])
    cmap_norm_ca = colors.Normalize(vmin=img_ca.min(), vmax=img_ca.max())
    img_unaligned_split_R = ax_unaligned_split_R.imshow(img_ca, cmap=cmap_ca, norm=cmap_norm_ca)

    ax_aligned = fig_alignment.add_subplot(gs0[1])
    alpha_aligned = 0.9
    img_aligned_L = ax_aligned.imshow(img_dual_aligned[:, :, 0], cmap=cmap_vm, norm=cmap_norm_vm,
                                      alpha=alpha_aligned, interpolation='bilinear')
    img_aligned_R = ax_aligned.imshow(img_dual_aligned[:, :, 1], cmap=cmap_ca, norm=cmap_norm_ca,
                                      alpha=alpha_aligned/2, interpolation='bilinear')

    for ax in [ax_unaligned_full, ax_unaligned_split_L, ax_unaligned_split_R, ax_aligned]:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    fig_alignment.savefig('Alignment_Vm-Ca.png')
    fig_alignment.show()

    # Save aligned output
    description = 'cv2 Alignment, ECC gradient\n' + im_filename + '\n'
    imsave('im_aligned_Left_Vm.tif', img_dual_aligned[:, :, 0], description=description + "Left/RH237/Vm")
    imsave('im_aligned_Right_Ca.tif', img_dual_aligned[:, :, 1], description=description + "Right/Rhod2/Ca")
    # cv2.imshow("Color Image", im)
    # cv2.imshow("Aligned Image", im_aligned[:, :, 0])
    cv2.waitKey(0)
