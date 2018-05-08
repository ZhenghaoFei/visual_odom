## Stereo Visual Odometry

This repository is C++ OpenCV implementation of SOFT (Stereo Odometry based on careful Feature selection and Tracking)

Original Paper: https://lamor.fer.hr/images/50020776/Cvisic2017.pdf

![alt text]([Features]:https://github.com/ZhenghaoFei/visual_odom/tree/master/images/features.png "features")

![alt text]([Trajectory]:https://github.com/ZhenghaoFei/visual_odom/tree/master/images/trajectory.png "trajectory")

### Requirements
[OpenCV 3.0](https://opencv.org/)

[Eigen 3.34](https://eigen.tuxfamily.org/dox/GettingStarted.html)

### Dataset
Tested on [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) odometry dataset

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
1. [Monocular visual odometry algorithm](https://github.com/avisingh599/mono-vo/blob/master/README.md)

2. [Matlab implementation of SOFT](https://github.com/Mayankm96/Stereo-Odometry-SOFT/blob/master/README.md)
