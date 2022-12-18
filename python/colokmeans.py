
import cv2
import numpy as np
import sys
image = cv2.imread(sys.argv[1])
image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
image = cv2.blur(image,(2,2))
# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# number of clusters (K)
k = 3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

print(centers)
#print(cv2.cvtColor(centers, cv2.COLOR_LAB2RGB))
# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()


# convert all pixels to the color of the centroids
segmented_image = cv2.cvtColor(centers[labels].reshape(image.shape), cv2.COLOR_YCrCb2BGR)
cv2.imshow('img', segmented_image)
cv2.waitKey()

for cluster in range(k):
    # disable only the cluster number 2 (turn the pixel into black)
    masked_image = np.copy(image)
    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to disable
    masked_image[labels == cluster] = [0, 0, 0]
    # convert back to original shape
    masked_image = masked_image.reshape(image.shape)
    # show the image
    cv2.imshow('img', masked_image)
    cv2.waitKey()
cv2.destroyAllWindows()