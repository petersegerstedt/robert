

def read_qr_code(filename):
    """Read an image and read the QR code.
    
    Args:
        filename (string): Path to file
    
    Returns:
        qr (string): Value from QR code
    """
    import cv2 as cv
    import numpy as np
    W = 1920
    H = 1080
    dst = dict(zip(('2', '3', '1', 'ul'), np.array([[[   2.    ,  886.    ],
        [ 192.    ,  886.    ],
        [ 192.    , 1076.    ],
        [   2.    , 1076.    ]],

       [[1724.    ,  885.    ],
        [1913.9999,  885.    ],
        [1913.9999, 1076.    ],
        [1724.    , 1076.    ]],

       [[1727.    ,    3.    ],
        [1916.9999,    3.    ],
        [1916.9999,  193.    ],
        [1727.    ,  193.    ]],

       [[   2.    ,    2.    ],
        [ 192.    ,    2.    ],
        [ 192.    ,  192.    ],
        [   2.    ,  192.    ]]], dtype=np.float32)))
    img = cv.imread(filename)
    detect = cv.QRCodeDetector()
    status, data, pts, _ = detect.detectAndDecodeMulti(img)
    print(data)

    if status:
        matched = [(k,src,dst[k]) for k,src in zip(data, pts) if k in dst]
        if len(matched):
            #pts1 = np.vstack([matched[0][1][:3],matched[1][1][0]])
            #pts2 = np.vstack([matched[0][2][:3],matched[1][2][0]])
            pts1 = np.vstack([pts for _,pts,__ in matched])
            pts2 = np.vstack([pts for _,__,pts in matched])
            matrix, _ = cv.findHomography(pts1, pts2)
            result = cv.warpPerspective(img, matrix, (W, H))
            roi = result[181:181+713, 176:176+1561]
            roi = cv.flip(roi, 0) # flip vertially
            roi = cv.flip(roi, 1) # flip horisontally
            cv.imwrite(filename + '.roi.jpg', roi)
            status2, data2, pts22, _ = detect.detectAndDecodeMulti(result)
            print (roi.shape)
            '''
            print (matrix)
            pts1 = matched[1][1]
            pts2 = matched[1][2]
            matrix = cv.getPerspectiveTransform(pts1, pts2)
            print (matrix)
            176 181 1561 713
            cropped_image = img[80:280, 150:330]
 
# Display cropped image
cv2.imshow("cropped", cropped_image)
 
# Save the cropped image
cv2.imwrite("Cropped Image.jpg", cropped_image)
            '''
            #blurred = cv.GaussianBlur(roi, (31, 31), 0)
            blurred = cv.blur(roi, (39, 39))
            grain_extracted = roi - blurred + 128
            #hist = cv.calcHist([grain_extracted],[0],None,[256],[0,256])
            from matplotlib import pyplot as plt
            color = ('b','g','r')
            for i,col in enumerate(color):
                histr = cv.calcHist([grain_extracted],[i],None,[256],[0,256])
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.savefig(filename + '.roi_histogram.png')
            cv.imwrite(filename + '.roi_grain_extract.jpg', grain_extracted)
            #cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
            #cv.imshow('image', grain_extracted)
            #cv.waitKey()
    return
    value, pts1, straight_qrcode = detect.detectAndDecode(img)
    print(value, len(pts1))
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