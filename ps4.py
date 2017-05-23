

import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

images = glob.glob( os.path.join('.','camera_cal','calibration*.jpg'))


objpoints = []
imgpoints = []

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

for fname in images:
    img = mpimg.imread(fname)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        # plt.imshow(img)
        # plt.show()


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

img = cv2.imread(os.path.join('.','camera_cal','calibration1.jpg'))

dst = cv2.undistort(img,mtx,dist,None,mtx)
plt.imshow(dst)
plt.show()




# detect and draw a single image's corners
# img = cv2.imread(os.path.join('.','camera_cal','calibration2.jpg'))
# objpoints = []
# imgpoints = []
#
# objp = np.zeros((6*9,3), np.float32)
# objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
#
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
#
# if ret == True:
#     imgpoints.append(corners)
#     objpoints.append(objp)
#
#     img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
#     plt.imshow(img)
#     plt.show()





















