# Chapter 1

## Raspberry OpenCV

https://pimylifeup.com/raspberry-pi-opencv/
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D ENABLE_NEON=ON \
    -D ENABLE_VFPV3=ON \
    -D BUILD_TESTS=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D CMAKE_SHARED_LINKER_FLAGS=-latomic \
    -D OPENCV_PYTHON_INSTALL_PATH=lib/python3.7/ \
    -D BUILD_EXAMPLES=OFF ..
```
