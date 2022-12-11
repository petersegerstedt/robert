
import cv2
from datetime import datetime
import time

def tictoc(msg, tic=time.perf_counter()):
    toc = time.perf_counter()
    print(toc-tic, msg)
    return toc

class CvVideoCapture:
    def __init__(self):
        self.tic = tictoc('CvVideoCapture initialized')
print('backends', cv2.videoio_registry.getBackends())
print('camerabackends', cv2.videoio_registry.getCameraBackends())


frameWidth = 640*2
frameHeight = 480*2
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)
tic = tictoc('started')
while cap.isOpened():
    success, img = cap.read()
    if success:
        tic = tictoc('image', tic)