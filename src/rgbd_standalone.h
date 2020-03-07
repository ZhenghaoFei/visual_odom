#ifndef RGBD_STANDALONE_H
#define RGBD_STANDALONE_H

#include <opencv2/opencv.hpp>
#include "camera_object.h"

struct v4l2_buffer;

class Intel_V4L2 : public CameraBase {
    public:
                        Intel_V4L2      ();
        virtual        ~Intel_V4L2      ();
                int     init_device     ();
                int     capture_frame   ();
        virtual void    getLRFrames     (cv::Mat &left_rect, cv::Mat &right_rect);
	    virtual void    getLRFrames     (unsigned char *left_rect, unsigned char *right_rect);

    private:
        int             m_fd;
        v4l2_buffer    *m_pV4l2Buffer;
        int             m_frameCount;
        int             m_bigBufferLength;
        unsigned char  *m_pBigBuffer;
        int             m_bufType;
};

#endif // RGBD_STANDALONE_H
