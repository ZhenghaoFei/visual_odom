## Stereo Visual Odometry

This repository is C++ OpenCV implementation of SOFT (Stereo Odometry based on careful Feature selection and Tracking)

Original Paper: https://lamor.fer.hr/images/50020776/Cvisic2017.pdf

### Requirements
[OpenCV 3.0](https://opencv.org/)

[Eigen 3.34](https://eigen.tuxfamily.org/dox/GettingStarted.html)

### Compile & Run
```bash
git clone https://github.com/ZhenghaoFei/visual_odom.git
```
Change **EIGENPATH** in visual_odometry/CMakeLists.txt

Change **filename_pose** & **filepath** in visual_odometry/src/main.cpp to your location of KITTI odom data

Change **projection matrixs** in visual_odometry/src/main.cpp to your calibration data

```bash
mkdir build
cd build
cmake ..
make -j4
./run
```
### Reference code
[Monocular visual odometry algorithm](https://github.com/avisingh599/mono-vo/blob/master/README.md)
[Matlab implementation of SOFT](https://github.com/Mayankm96/Stereo-Odometry-SOFT/blob/master/README.md)
