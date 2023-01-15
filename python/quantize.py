from palettes import jasc
import numpy as np
import cv2

def quantize_to_palette(image, palette):
    X_query = image.reshape(-1, 3).astype(np.float32)
    X_index = palette.astype(np.float32)

    knn = cv2.ml.KNearest_create()
    knn.train(X_index, cv2.ml.ROW_SAMPLE, np.arange(len(palette)))
    ret, results, neighbours, dist = knn.findNearest(X_query, 1)

    quantized_image = np.array([palette[idx] for idx in neighbours.astype(int)])
    quantized_image = quantized_image.reshape(image.shape)
    return quantized_image.astype(np.uint8)

import sys

palette = np.array(jasc.loads(open(sys.argv[1], 'r').read()))
image = cv2.imread(sys.argv[2])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = quantize_to_palette(image, palette)
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
print('input', image.shape, image.dtype)
print('result', result.shape , result.dtype)

cv2.imshow('img', image)
cv2.waitKey()
cv2.imshow('img', result)
cv2.waitKey()
cv2.destroyAllWindows()