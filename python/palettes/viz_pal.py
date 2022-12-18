import jasc
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import numpy as np
import cv2


if '__main__' == __name__:
    import sys
    
    with open(sys.argv[1], 'r') as f:
        pal = jasc.load(f)
        entries = [np.full((10, 10, 3), e, dtype=np.uint8) / 255.0 for e in pal]
        rows_cols = int(len(entries)**0.5)
        for i, e in enumerate(entries):
            h,s,v = rgb_to_hsv(pal[i])
            title = ','.join(f'{x:.2f}' for x in [h,s])
            title = str(int(v))
            plt.subplot(rows_cols, rows_cols, i+1, title = title)
            plt.imshow(e)
        plt.savefig(f'{sys.argv[1]}.png')
        
        light_red = rgb_to_hsv(pal[15])
        dark_red = rgb_to_hsv(pal[12])
        light_green = rgb_to_hsv(pal[8])
        dark_green = rgb_to_hsv(pal[14])
        white = rgb_to_hsv((255,255,255))
        black = rgb_to_hsv((0,0,0))
        
        im = cv2.imread(sys.argv[2])
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(im_hsv, light_red, dark_red)
        result_red = cv2.bitwise_and(im, im, mask=red_mask)
        cv2.imshow('img', result_red)
        cv2.waitKey()
        green_mask = cv2.inRange(im_hsv, light_green, dark_green)
        cv2.imshow('img', green_mask)
        cv2.waitKey()
        dummy = cv2.inRange(im_hsv, white, black)
        cv2.imshow('img', dummy)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
"""
 plt.subplot(1, 2, 1)
>>> plt.imshow(hsv_to_rgb(do_square))
>>> plt.subplot(1, 2, 2)
>>> plt.imshow(hsv_to_rgb(lo_square))
>>> plt.show()
"""