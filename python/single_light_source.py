import random
import cv2
import numpy as np
import scipy.optimize as opt

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Find a parabolic function that fits the image background')
    parser.add_argument('image', type=str, help='input image', metavar='FILE')
    parser.add_argument('--cols', default=30, type=int, help='nbr of sample columns')
    parser.add_argument('--rows', default=30, type=int, help='nbr of sample rows')
    parser.add_argument('--nodata', default=255, type=int, help='nbr of sample rows')
    args = parser.parse_args()
    im = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    h,w = im.shape[:2]
    pts = np.array([[x,y,im[y][x]] for x in range(w//(args.cols*2), w, w//args.cols) for y in range(h//(args.rows*2), h, h//args.rows) if args.nodata != im[y][x]])
    print(pts.shape)
    X,Y,Z = pts.swapaxes(0,1)
    #print(*x)
    #print(*y)
    #print(*z)
    imax = np.argmax(Z)
    zmax = Z.max()
    print(X[imax], Y[imax], zmax)
    def parabol_func(data,a,b,c,d):
        x,y = data
        return -(((x-b)/a)**2+((y-d)/c)**2)+zmax
    popt,pcov=opt.curve_fit(parabol_func,(X,Y),Z,p0=[zmax,X[imax],zmax,Y[imax]])
    print(popt)
    im2 = np.array((-parabol_func(np.indices(im.shape),*popt)+255).round(), dtype=np.uint8)
    print(im2.shape)
    cv2.imwrite(args.image + '.bg.png', im2)
    im3 = cv2.add(im, im2)
    '''
    im3 = cv2.cvtColor(cv2.add(im, im2), cv2.COLOR_GRAY2RGB)
    for x,y,z in pts:
        cv2.circle(im3, (x,y), 3, (255,0,0))
    '''
    cv2.imwrite(args.image + '.bg_extract.png', im3)
'''
>>> def paraBolEqn(data,a,b,c,d):
...     x,y = data
...     return -(((x-b)/a)**2+((y-d)/c)**2)+185
...
>>> 
>>> popt
array([ 90.27508524, 559.38652775,  76.49329964, 621.91322357])
>>> 
'''