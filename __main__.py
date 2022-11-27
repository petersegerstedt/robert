def read_qr_code(filename):
    """Read an image and read the QR code.
    
    Args:
        filename (string): Path to file
    
    Returns:
        qr (string): Value from QR code
    """
    import cv2 as cv
    import numpy as np
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    detect = cv.QRCodeDetector()
    value, pts1, straight_qrcode = detect.detectAndDecode(img)
    print(value)
    print(pts1[0])
    colors = [(255, 0, 0),(0, 255, 0),(0, 0, 255),(0, 255, 255)]
    for i, xy in enumerate(pts1[0]):
        x, y = [int(f) for f in xy]
        print(i,x,y,colors[i])
        img = cv.circle(img, (x,y) , radius=3, color=colors[i], thickness=1)
    '''

    qr_size = 50
    w,h = (800,800)
    pts2 = np.float32([[0,0],[1,0],[1,1],[0,1]]) * qr_size + [w-qr_size,h-qr_size]
    print(pts2)
    matrix = cv.getPerspectiveTransform(pts1[0], pts2)
    result = cv.warpPerspective(img, matrix, (w, h))

    minHessian = 400
detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
keypoints = detector.detect(src)
    '''
    qr_size = 60
    w,h = (800, 800)
    margin = 80
    pts2 = np.float32([[0,0],[1,0],[1,1],[0,1]]) * qr_size + np.float32([w-qr_size-margin,h-qr_size-margin])
    #print(pts2)
    #pts2 = np.float32([[960, 448], [1024, 448],[1024, 512], [960, 512]])
    print(pts2)

    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # draw only keypoints location,not size and orientation
    img = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    #plt.imshow(img2), plt.show()

    matrix = cv.getPerspectiveTransform(pts1[0], pts2)
    result = cv.warpPerspective(img, matrix, (w, h))
    cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
    cv.imshow('image', result)
    cv.waitKey()
    return value
if __name__ == '__main__':
    import sys
    sys.exit(read_qr_code(*sys.argv[1:]))