#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;


int main(int argc, char* argv[])
{
	int fbfd = 0;
	char *fbp = 0;
	const int sizeInBytes = 1280*720*4;
	
	// Open the file for reading and writing
  fbfd = open("/dev/fb0", O_RDWR);
  

  fbp = (char*)mmap(0, 
                    sizeInBytes, 
                    PROT_READ | PROT_WRITE, 
                    MAP_SHARED, 
                    fbfd, 0);
	
	VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	uchar *data;
	Mat frame, frame2;
	while(true)
	{
		cap >> frame;
                //process the frame...
		cvtColor(frame, frame2, COLOR_BGR2BGRA);
		data = frame2.data;
		if ((int)fbp == -1) {
		printf("Failed to mmap.\n");
		}
		else {
		memcpy(fbp, data, sizeInBytes);
		}
	}

  // cleanup
  munmap(fbp, sizeInBytes);
  close(fbfd);
  return 0;
}