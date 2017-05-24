

import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from moviepy.editor import VideoFileClip

# images = glob.glob( os.path.join('.','camera_cal','calibration*.jpg'))
#
#
# objpoints = []
# imgpoints = []
#
# objp = np.zeros((6*9,3), np.float32)
# objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
#
# for fname in images:
#     img = mpimg.imread(fname)
#
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
#     ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
#
#     if ret == True:
#         imgpoints.append(corners)
#         objpoints.append(objp)
#
#         img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
#         # plt.imshow(img)
#         # plt.show()
#
#
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
#
# img = cv2.imread(os.path.join('.','camera_cal','calibration1.jpg'))
#
# dst = cv2.undistort(img,mtx,dist,None,mtx)
# # plt.imshow(dst)
# # plt.show()
#
#
# # detect and draw a single image's corners
# # img = cv2.imread(os.path.join('.','camera_cal','calibration2.jpg'))
# # objpoints = []
# # imgpoints = []
# #
# # objp = np.zeros((6*9,3), np.float32)
# # objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# #
# # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #
# # ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
# #
# # if ret == True:
# #     imgpoints.append(corners)
# #     objpoints.append(objp)
# #
# #     img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
# #     plt.imshow(img)
# #     plt.show()
#
#
# ############## Part 2 Piplining
#
# def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     # Take both Sobel x and y gradients
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
#     # Calculate the gradient magnitude
#     gradmag = np.sqrt(sobelx**2 + sobely**2)
#     # Rescale to 8 bit
#     scale_factor = np.max(gradmag)/255
#     gradmag = (gradmag/scale_factor).astype(np.uint8)
#     # Create a binary image of ones where threshold is met, zeros otherwise
#     binary_output = np.zeros_like(gradmag)
#     binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
#
#     # Return the binary image
#     return binary_output
#
# # Define a function to threshold an image for a given range and Sobel kernel
# def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
#     # Grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     # Calculate the x and y gradients
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
#     # Take the absolute value of the gradient direction,
#     # apply a threshold, and create a binary image result
#     absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
#     binary_output =  np.zeros_like(absgraddir)
#     binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
#
#     # Return the binary image
#     return binary_output
#
# def my_abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
#     def scale_sobel(input):
#         abs_sobelx = np.absolute(input)
#         return np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
#
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
#     scaled_sobel = scale_sobel(sobelx) if orient == 'x' else scale_sobel(sobely)
#     sxbinary = np.zeros_like(scaled_sobel)
#     sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
#     # plt.imshow(sxbinary, cmap='gray')
#     return sxbinary
#
# def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
#     thresh_min = thresh[0]
#     thresh_max = thresh[1]
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     # Apply x or y gradient with the OpenCV Sobel() function
#     # and take the absolute value
#     if orient == 'x':
#         abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel))
#     if orient == 'y':
#         abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel))
#     # Rescale back to 8 bit integer
#     scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
#     # Create a copy and apply the threshold
#     binary_output = np.zeros_like(scaled_sobel)
#     # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
#     binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
#
#     # Return the result
#     return binary_output
#
# def get_combined_gradient(img):
#     # Choose a Sobel kernel size
#     ksize = 3  # Choose a larger odd number to smooth gradient measurements
#
#     # Apply each of the thresholding functions
#     gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(100, 255))
#     grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(100, 255))
#     mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(50, 255))
#     dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))
#     dir_binary2 = np.uint8(255 * dir_binary / np.max(dir_binary))
#     # plt.imshow(dir_binary2)
#     # plt.show()
#     combined = np.zeros_like(dir_binary)
#     combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
#     combined = np.uint8(255 * combined / np.max(combined))
#
#     return  combined
#
# def get_combined(img):
#     # Convert to HLS color space and separate the S channel
#     # Note: img is the undistorted image
#     hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     s_channel = hls[:, :, 2]
#
#     # Grayscale image
#     # NOTE: we already saw that standard grayscaling lost color information for the lane lines
#     # Explore gradients in other colors spaces / color channels to see what might work better
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#
#     # Sobel x
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
#     abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
#     scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
#
#     # Threshold x gradient
#     thresh_min = 20
#     thresh_max = 100
#     sxbinary = np.zeros_like(scaled_sobel)
#     sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
#
#     sxbinary = get_combined_gradient(img)
#
#     # Threshold color channel
#     s_thresh_min = 170
#     s_thresh_max = 255
#     s_binary = np.zeros_like(s_channel)
#     s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
#
#
#
#     # Stack each channel to view their individual contributions in green and blue respectively
#     # This returns a stack of the two binary images, whose components you can see as different colors
#     color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
#
#     # Combine the two binary thresholds
#     combined_binary = np.zeros_like(sxbinary)
#     combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
#     # plt.imshow(combined_binary)
#     # plt.show()
#     return combined_binary
#
#
# def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
#     combined_binary = get_combined(img)
#     return combined_binary
#
# def process_image(img):
#     img = cv2.undistort(img,mtx,dist,None,mtx)
#     result = pipeline(img)
#     return np.uint8(result*255.0)
#
# # white_output = 'test_videos_output/solidWhiteRight.mp4'
# #
# # clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
# # white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# # white_clip.write_videofile(white_output, audio=False)
#
# test_image_location = ".\\test_images\\straight_lines1.jpg"
# output_image_location = ".\\output_images\\output1.jpg"
#
# test_image = cv2.imread(test_image_location)
# output_image = process_image(test_image)
# # plt.imshow(output_image)
# # plt.show()
# cv2.imwrite(output_image_location,output_image)


images = glob.glob('./test_images/*.jpg')

for i in range(len(images)):
    img = cv2.imread(images[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # top_left = [562,470]
    # top_right = [721,470]
    top_left = [560, 470]
    top_right = [720, 470]
    bottom_right = [950,620]
    bottom_left = [350,620]
    pts = np.array([bottom_left,bottom_right,top_right,top_left])

    # plt.imshow(img)
    # plt.plot(top_left[0],top_left[1],'.')
    # plt.plot(top_right[0], top_right[1], '.')
    # plt.plot(bottom_right[0], bottom_right[1], '.')
    # plt.plot(bottom_left[0], bottom_left[1], '.')
    # plt.show()

    top_left_dst = [320,0]
    top_right_dst = [980,0]
    bottom_right_dst = [980,720]
    bottom_left_dst = [320,720]
    dst_pts = np.array([bottom_left_dst, bottom_right_dst, top_right_dst, top_left_dst])

    ## you can plot this and see that lines are properly chosen
    ## cv2.polylines(img,[pts],True,(255,0,0), 5)
    ## cv2.polylines(img,[dst_pts],True,(0,0,255), 5)

    pts = np.float32(pts.tolist())
    dst_pts = np.float32(dst_pts.tolist())

    ## M is the transformation matrix
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, pts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 15))
    image_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, dsize = image_size, flags = cv2.INTER_LINEAR)
    ax1.imshow(img)
    ax1.plot(top_left[0], top_left[1], '.')
    ax1.plot(top_right[0], top_right[1], '.')
    ax1.plot(bottom_right[0], bottom_right[1], '.')
    ax1.plot(bottom_left[0], bottom_left[1], '.')
    # ax2.imshow(warped)
    # plt.show()

    # Read in a thresholded image
    # warped = mpimg.imread('warped_example.jpg')
    warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
    # window settings
    window_width = 50
    window_height = 80  # Break image into 9 vertical layers since image height is 720
    margin = 100  # How much to slide left and right for searching


    def window_mask(width, height, img_ref, center, level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
        max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
        return output


    def find_window_centroids(image, window_width, window_height, margin):

        window_centroids = []  # Store the (left,right) window centroid positions per level
        window = np.ones(window_width)  # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(warped.shape[0] / window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(
                warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height),
                :], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width / 2
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            # Add what we found for that layer
            window_centroids.append((l_center, r_center))

        return window_centroids


    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.array(cv2.merge((warped, warped, warped)),
                           np.uint8)  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    # Display the final results
    plt.imshow(output)
    plt.title('window fitting results')
    plt.show()









    
    # Generate some fake data to represent lane-line pixels
    ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
    quadratic_coeff = 3e-4  # arbitrary quadratic coefficient
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = l_points
    rightx = r_points


    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Plot up the fake data
    mark_size = 3
    plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
    plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, ploty, color='green', linewidth=3)
    plt.plot(right_fitx, ploty, color='green', linewidth=3)
    plt.gca().invert_yaxis()  # to visualize as we do the images









    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.imshow(result)
    plt.show()


    def draw_polygon(image, left_coeffs, right_coeffs):
        left_points = []
        right_points = []

        left_eq = np.poly1d(left_coeffs)
        right_eq = np.poly1d(right_coeffs)

        for y in range(720):
            x = int(left_eq(y))
            if x < 0:
                pass
            else:
                left_points.append([x, y])

        for y in range(720):
            x = int(right_eq(y))
            if x < 0:
                pass
            else:
                right_points.append([x, y])

        polygon_points = left_points + right_points[::-1]

        final_image = np.zeros_like(image)
        final_image = cv2.fillPoly(final_image, [np.array(polygon_points)], color=(102, 255, 102))










