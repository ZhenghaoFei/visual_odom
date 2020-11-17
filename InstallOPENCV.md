#### Install CUDA
#### Comiple CUDA OPENCV && OPENCV_contrib
```bash
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libgtk2.0-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install libhdf5-serial-dev
sudo apt-get install python2.7-dev

version=4.5.0
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout $version
opencv_contrib_path=$PWD/modules/
cd ..

git clone https://github.com/opencv/opencv.git
cd opencv
git checkout $version
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release     \
         -DCMAKE_INSTALL_PREFIX=/usr     \
         -DBUILD_PNG=OFF     \
         -DBUILD_TIFF=OFF     \
         -DBUILD_TBB=OFF     \
         -DBUILD_JPEG=OFF     \
         -DBUILD_JASPER=OFF     \
         -DBUILD_ZLIB=OFF     \
         -DBUILD_EXAMPLES=OFF     \
         -DBUILD_JAVA=OFF     \
         -DBUILD_opencv_python2=ON     \
         -DBUILD_opencv_python3=OFF     \
         -DWITH_OPENCL=OFF     \
         -DWITH_OPENMP=OFF     \
         -DWITH_FFMPEG=ON     \
         -DWITH_GSTREAMER=OFF     \
         -DWITH_GSTREAMER_0_10=OFF     \
         -DWITH_CUDA=ON     \
         -DWITH_GTK=ON     \
         -DWITH_VTK=OFF     \
         -DWITH_TBB=ON     \
         -DWITH_1394=OFF     \
         -DWITH_OPENEXR=OFF    \
         -DINSTALL_C_EXAMPLES=OFF     \
         -DINSTALL_TESTS=OFF  \
         -DWITH_CUDA=ON \
         -DOPENCV_EXTRA_MODULES_PATH=$opencv_contrib_path
make -j4
sudo make install
```