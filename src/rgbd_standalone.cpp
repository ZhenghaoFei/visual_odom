// See example codes at https://linuxtv.org/downloads/v4l-dvb-apis/uapi/v4l/capture.c.html
// and https://gist.github.com/mike168m/6dd4eb42b2ec906e064d

#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <fcntl.h>
#include <unistd.h>

#include <linux/ioctl.h>
#include <linux/types.h>
#include <linux/v4l2-common.h>
#include <linux/v4l2-controls.h>
#include <linux/videodev2.h>

#include <sys/ioctl.h>
#include <sys/mman.h>

#include <string.h>
#include <fstream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include "rgbd_standalone.h"

Intel_V4L2::Intel_V4L2 () 
    : CameraBase ()
{
    m_bigBufferLength = 0;
    m_frameCount      = 0;
    m_fd              = -1;
    m_pBigBuffer      = NULL;
    m_pV4l2Buffer     = 0;
    init_device ();
}

Intel_V4L2::~Intel_V4L2 () {
    if (NULL != m_pV4l2Buffer)
        free (m_pV4l2Buffer);
    if (NULL != m_pBigBuffer)
        free (m_pBigBuffer);
    
    if (0 > m_fd) return;

    if (0 > ioctl (m_fd, VIDIOC_STREAMOFF, &m_bufType)) {
        perror ("Intel_V4L2::~Intel_V4L2 : VIDIOC_STREAMOFF : ");
    }

    close (m_fd);
}

int Intel_V4L2::init_device () {

    // Intel RGBD ir frames @/dev/video1 (Interlaced left and right)
    m_fd = open ("/dev/video1", O_RDWR);
    if (0 > m_fd) {
        perror ("Intel_V4L2::init_device : failed to open /dev/video1");
        return 1;
    }

    struct v4l2_capability cap;
    if (0 > ioctl (m_fd, VIDIOC_QUERYCAP, &cap)) {
        perror ("Intel_V4L2::init_device : VIDIOC_QUERYCAP : ");
        return 1;
    }

    // Image format parameters :
    struct v4l2_format fmt;
    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = 640;
    fmt.fmt.pix.height      = 480;
    fmt.fmt.pix.pixelformat = v4l2_fourcc('Y', '8', 'I', ' ');
    fmt.fmt.pix.field       = V4L2_FIELD_NONE;
    if (0 > ioctl (m_fd, VIDIOC_S_FMT, &fmt)) {
        perror ("Intel_V4L2::init_device : VIDIOC_S_FMT : ");
        return 1;
    }

    // Setup buffers. For RGBD, need to use memory buffer obtained from malloc.
    struct v4l2_requestbuffers req = {0};
    req.count   = 1;
    req.type    = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    // req.memory = V4L2_MEMORY_MMAP;
    req.memory = V4L2_MEMORY_USERPTR;

    if (0 > ioctl (m_fd, VIDIOC_REQBUFS, &req)) {
        perror ("Intel_V4L2::init_device : VIDIOC_REQBUFS : ");
        return 1;
    }

    // Set Frame-rate. Need to be very slow.
    struct v4l2_streamparm streamParm = {0};
    // struct v4l2_captureparm
    // capability=V4L2_CAP_TIMEPERFRAME, capturemode=0, timeperframe=1/6, extendedmode=0, readbuffers=0
    streamParm.type                      = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    streamParm.parm.capture.capability   = V4L2_CAP_TIMEPERFRAME;
    streamParm.parm.capture.capturemode  = 0;
    streamParm.parm.capture.timeperframe.numerator   = 1;
    streamParm.parm.capture.timeperframe.denominator = 6;
    streamParm.parm.capture.extendedmode = 0;
    streamParm.parm.capture.readbuffers  = 0;
    if (0 > ioctl (m_fd, VIDIOC_S_PARM, &streamParm)) {
        perror ("Intel_V4L2::init_device : VIDIOC_S_PARM : ");
        return 1;
    }

    v4l2_buffer buf = {0};
    buf.type    = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory  = V4L2_MEMORY_USERPTR;
    buf.index   = 0;
    if (0 > ioctl (m_fd, VIDIOC_QUERYBUF, &buf)) {
        perror ("Intel_V4L2::init_device : VIDIOC_QUERYBUF : ");
        return 1;
    }

    // Allocate a big buffer once.
    m_bigBufferLength = 1280 * 720 * 4 * 2 + 1024;
    m_pBigBuffer = (unsigned char *) malloc (m_bigBufferLength);
    assert (NULL != m_pBigBuffer);
    memset (m_pBigBuffer, 0, m_bigBufferLength);

    m_pV4l2Buffer = (v4l2_buffer *) calloc (1, sizeof (v4l2_buffer));
    m_pV4l2Buffer->type       = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    m_pV4l2Buffer->memory     = V4L2_MEMORY_USERPTR;
    m_pV4l2Buffer->index      = 0;
    m_pV4l2Buffer->m.userptr  = (unsigned long) m_pBigBuffer;
    m_pV4l2Buffer->length     = m_bigBufferLength;

    // USB-UVC streaming:
    m_bufType = m_pV4l2Buffer->type;
    if (0 > ioctl (m_fd, VIDIOC_STREAMON, &m_bufType)) {
        perror ("Intel_V4L2::init_device : VIDIOC_STREAMON");
        return 1;
    }

    return 0;
}

int Intel_V4L2::capture_frame () {

    usleep (100000);

    if (0 > ioctl (m_fd, VIDIOC_QBUF, m_pV4l2Buffer)) {
        perror ("Intel_V4L2::capture_frame : VIDIOC_QBUF : ");
        return -1;
    }

    usleep (100000);

    if (0 > ioctl (m_fd, VIDIOC_DQBUF, m_pV4l2Buffer)) {
        perror ("Intel_V4L2::capture_frame : VIDIOC_DQBUF : ");
        return -1;
    }

    std::cout << "Intel_V4L2::capture_frame : received bytes [" << m_pV4l2Buffer->bytesused << "]" << std::endl;
    printf ("Intel_V4L2::capture_frame : flags=[0x%08x]\n", m_pV4l2Buffer->flags);

    #ifdef STANDALONE_TEST
    if (m_frameCount % 10 == 0) {
      char buf[1024];
      sprintf (buf, "raw_data%02d.bin", m_frameCount);
      FILE *pFout = fopen (buf, "w");
      // size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);
      fwrite (m_pBigBuffer, m_pV4l2Buffer->bytesused, 1, pFout);
      fclose (pFout);
    }
    #endif
    m_frameCount++;
    usleep (10000);
    return 0;
}

void Intel_V4L2::getLRFrames (cv::Mat &left_rect, cv::Mat &right_rect) {
    int status = capture_frame ();
    if (0 != status) {
        std::cout << "Error : failed to capture Infrared frame." << std::endl;
        return;
    }
    left_rect.create  (480, 640, CV_8UC1);
    right_rect.create (480, 640, CV_8UC1);
    unsigned short *interlacedBuffer = (unsigned short *) m_pBigBuffer;
    for (int i = 0; i < 480; i++) {
        for (int j = 0; j < 640; j++) {
            unsigned short lr_pixel = interlacedBuffer[i*640+j];
            left_rect.at<unsigned char> (i, j)  = lr_pixel % 256;
            right_rect.at<unsigned char> (i, j) = lr_pixel / 256;
        }
    }
    incrementFrameCount ();
    saveFrame (left_rect, right_rect);
}

void Intel_V4L2::getLRFrames (unsigned char *left_rect, unsigned char *right_rect) {
    int status = capture_frame ();
    if (0 != status) {
        std::cout << "Error : failed to capture Infrared frame." << std::endl;
        return;
    }
    unsigned short *interlacedBuffer = (unsigned short *) m_pBigBuffer;
    for (int i = 0; i < 480; i++) {
        for (int j = 0; j < 640; j++) {
            unsigned short lr_pixel = interlacedBuffer[i*640+j];
            left_rect[i*640+j]  = lr_pixel % 256;
            right_rect[i*640+j] = lr_pixel / 256;
        }
    }

    #if 1
    cv::Mat leftMat  = cv::Mat::zeros (480, 640, CV_8U);
    cv::Mat rightMat = cv::Mat::zeros (480, 640, CV_8U);

    for (int i = 0; i < 480; i++) {
        for (int j = 0; j < 640; j++) {
            leftMat.at<uchar>(i, j)  = left_rect[i*640+j];
            rightMat.at<uchar>(i, j) = right_rect[i*640+j];
        }
    }

    saveFrame (leftMat, rightMat);

    incrementFrameCount ();
    #endif
}

#ifdef STANDALONE_TEST
int main(int argc, char *argv[]) {
    Intel_V4L2 intelObject;
    for (int i = 0; i < 20; i++)
        intelObject.capture_frame ();
}
#endif
