<<<<<<< HEAD
import cv2
import numpy as np
import os
import glob
def save_coefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()
def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]

def main(img_glob_pat):

    # Defining the dimensions of checkerboard
    CHECKERBOARD = (8,5)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
    # Extracting path of individual image stored in a given directory
    images = glob.glob(img_glob_pat)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        else:
            print('corners not found', fname)
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
    
    h,w = img.shape[:2]
    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    #print("rvecs : \n")
    #print(rvecs)
    #print("tvecs : \n")
    #print(tvecs)
    ''' TODO find out more about this:...probably most useful for stereo vision...
    # getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha[, newImgSize[, centerPrincipalPoint]]) -> retval, validPixROI
    newCameraMatrix, validPixROI = cv2.getOptimalNewCameraMatrix(mtx, dist, img.shape[:2], 1) # alpha(0..1.0) > 0 should keep all pixels, use when corners are useful
    # initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type[, map1[, map2]]) -> map1, map2
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, newCameraMatrix, img.shape[:2], cv2.CV_32FC1)
    print('debug:', map1, map2)
    remapped = cv2.remap(img, map1, map2, cv2.INTER_NEAREST)
    '''
    
    # cv2.undistort: 
    # The function is simply a combination of #initUndistortRectifyMap (with unity R ) and #remap (with bilinear interpolation). 
    # See the former function for details of the transformation being performed.
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imshow('img',undist)
    cv2.waitKey(0)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return mtx, dist
    
if __name__ == '__main__':
    import sys
    img_glob_pat = sys.argv[1]
    save_file = sys.argv[2]
    mtx, dist = main(img_glob_pat)
    save_coefficients(mtx, dist, save_file)
=======
import cv2
import numpy as np
import os
import glob
def save_coefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()
def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]

def main(img_glob_pat):

    # Defining the dimensions of checkerboard
    CHECKERBOARD = (8,5)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
    # Extracting path of individual image stored in a given directory
    images = glob.glob(img_glob_pat)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        else:
            print('corners not found', fname)
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
    
    h,w = img.shape[:2]
    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    #print("rvecs : \n")
    #print(rvecs)
    #print("tvecs : \n")
    #print(tvecs)
    ''' TODO find out more about this:...probably most useful for stereo vision...
    # getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha[, newImgSize[, centerPrincipalPoint]]) -> retval, validPixROI
    newCameraMatrix, validPixROI = cv2.getOptimalNewCameraMatrix(mtx, dist, img.shape[:2], 1) # alpha(0..1.0) > 0 should keep all pixels, use when corners are useful
    # initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type[, map1[, map2]]) -> map1, map2
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, newCameraMatrix, img.shape[:2], cv2.CV_32FC1)
    print('debug:', map1, map2)
    remapped = cv2.remap(img, map1, map2, cv2.INTER_NEAREST)
    '''
    
    # cv2.undistort: 
    # The function is simply a combination of #initUndistortRectifyMap (with unity R ) and #remap (with bilinear interpolation). 
    # See the former function for details of the transformation being performed.
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imshow('img',undist)
    cv2.waitKey(0)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return mtx, dist
    
if __name__ == '__main__':
    import sys
    img_glob_pat = sys.argv[1]
    save_file = sys.argv[2]
    mtx, dist = main(img_glob_pat)
    save_coefficients(mtx, dist, save_file)
>>>>>>> 1e7abed... work on 5CD0154ZXS
    sys.exit(0)