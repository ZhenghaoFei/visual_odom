#ifndef CAMERA_OBJECT_H
#define CAMERA_OBJECT_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>

class CameraBase {
    public:
                      CameraBase  () : m_saveFrames (true), m_frameCount (0) { m_saveFrames = (getenv ("SAVE_FRAMES")) ? true : false; }
      virtual        ~CameraBase  () {}
      virtual void    getLRFrames (unsigned char *left_rect, unsigned char *right_rect) = 0;
      virtual void    getLRFrames (cv::Mat &left_rect, cv::Mat &right_rect) = 0;
              void    saveRawFrame (cv::Mat &frame_left, cv::Mat &frame_right) {
                if (false == m_saveFrames) return;
                char fileName[1024];
                snprintf (fileName, sizeof (fileName), "images/raw_left_%d.bmp", m_frameCount);
                imwrite (fileName, frame_left);
                snprintf (fileName, sizeof (fileName), "images/raw_right_%d.bmp", m_frameCount);
                imwrite (fileName, frame_right);
              };
              void    saveFrame (cv::Mat &frame_left, cv::Mat &frame_right) {
                if (false == m_saveFrames) return;
                char fileName[1024];
                snprintf (fileName, sizeof (fileName), "images/left_%d.png", m_frameCount);
                imwrite (fileName, frame_left);
                snprintf (fileName, sizeof (fileName), "images/right_%d.png", m_frameCount);
                imwrite (fileName, frame_right);
                {
                    snprintf (fileName, sizeof (fileName), "images/left_%d.yml", m_frameCount);
                    cv::FileStorage fsl (fileName, cv::FileStorage::WRITE);
                    fsl << "image" << frame_left;
                    snprintf (fileName, sizeof (fileName), "images/right_%d.yml", m_frameCount);
                    cv::FileStorage fsr (fileName, cv::FileStorage::WRITE);
                    fsr << "image" << frame_right;
                }
              }
              void incrementFrameCount () {m_frameCount++; }
    protected:
        bool    m_saveFrames;
        int     m_frameCount;
};

#endif // CAMERA_OBJECT_H
