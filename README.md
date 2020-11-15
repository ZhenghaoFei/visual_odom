## Stereo Visual Odometry

This repository is C++ OpenCV implementation of Stereo Visual Odometry, using OpenCV calcOpticalFlowPyrLK for feature tracking.

Reference Paper: https://lamor.fer.hr/images/50020776/Cvisic2017.pdf

Demo vedio: https://www.youtube.com/watch?v=Z3S5J_BHQVw&t=17s

![alt text](https://github.com/ZhenghaoFei/visual_odom/blob/master/images/features.png "features")

![alt text](https://github.com/ZhenghaoFei/visual_odom/blob/master/images/trajectory.png "trajectory")

### Requirements
[OpenCV 3.0](https://opencv.org/)
```bash
sudo apt update
sudo apt install libopencv-dev 
```

### Dataset
Tested on [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) odometry dataset

### Compile & Run
```bash
git clone https://github.com/ZhenghaoFei/visual_odom.git
```
The system use **Camera Parameters** in calibration/xx.yaml, put your own camera parameters in the same format and pass the path when you run.

```bash
mkdir build
cd build
cmake ..
make -j4
./run /PathToKITTI/sequences/00/ ../calibration/kitti00.yaml
```

### GPU CUDA acceleration
Thanks to [temburuyk](https://github.com/ZhenghaoFei/visual_odom/commits?author=temburuyk), the most time consumtion function circularMatching() can be accelerated using CUDA and greately improve the performance.  
To enable GPU acceleration
1. Make sure you have CUDA compatible GPU.
2. Install CUDA, compile and install CUDA supported OpenCV 
3. When compiling, use 
```bash
cmake .. -DUSE_CUDA=on
```
4. Compile & Run

### Reference code
1. [Monocular visual odometry algorithm](https://github.com/avisingh599/mono-vo/blob/master/README.md)

2. [Matlab implementation of SOFT](https://github.com/Mayankm96/Stereo-Odometry-SOFT/blob/master/README.md)
