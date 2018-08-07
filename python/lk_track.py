
'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
# import video
# from common import anorm2, draw_str
from time import clock

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.3,
                       minDistance = 3,
                       blockSize = 3 )

def kp_to_pt(kps):
    np_pt = np.zeros((len(kps), 2))
    for i in range(len(kps)):
        np_pt[i, :] = kps[i].pt
    return np.float32(np_pt)

def featureDetectionFast(img_gary):
    fast = cv.FastFeatureDetector_create(100)
    kp = fast.detect(img_gary, None)
    return kp

def load_image(path):
    img_color = cv.imread(path)
    img_gary = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    return img_color, img_gary

class App:
    def __init__(self, video_src, t_init, steps):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.img_prefix = video_src
        self.frame_idx = 0
        self.t_init = t_init
        self.steps = steps

    def stereoTracking(self):
        valid_points = []
        for i in range(self.steps):
            t = self.t_init + i
            print('Time step: ', t)

            # image path
            img_id_t0 = t
            img_id_t1 = t + 1
            img_left_path_t0 = self.img_prefix + 'left/' + '%06d' % img_id_t0 +'.png'
            img_right_path_t0 = self.img_prefix + 'right/' + '%06d' % img_id_t0 +'.png'
            img_left_path_t1 = self.img_prefix + 'left/' + '%06d' % img_id_t1 +'.png'
            img_right_path_t1 = self.img_prefix + 'right/' + '%06d' % img_id_t1 +'.png'

            # left image t0
            img_left_color_t0, img_left_gary_t0 = load_image(img_left_path_t0)

            # right image t0 
            img_right_color_t0, img_right_gary_t0 = load_image(img_right_path_t0)

            # left image t1
            img_left_color_t1, img_left_gary_t1 = load_image(img_left_path_t1)

            # right image t1
            img_right_color_t1, img_right_gary_t1 = load_image(img_right_path_t1)

            vis = img_left_color_t0.copy()
            img_left_vis = img_left_color_t0.copy()
            img_right_vis = img_left_color_t0.copy()


            if len(self.tracks) > 0:
                kp_left_t0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)

                # kp_left_t1, st_0, err = cv.calcOpticalFlowPyrLK(img_left_gary_t0, img_left_gary_t1, kp_left_t0, None, None, **lk_params)
                # kp_left_t0r, st_1, err = cv.calcOpticalFlowPyrLK(img_left_gary_t1, img_left_gary_t0, kp_left_t1, None, None, **lk_params)

                kp_right_t0, st_0, err = cv.calcOpticalFlowPyrLK(img_left_gary_t0, img_right_gary_t0, kp_left_t0, None, None, **lk_params)
                kp_right_t1, st_1, err = cv.calcOpticalFlowPyrLK(img_right_gary_t0, img_right_gary_t0, kp_right_t0, None, None, **lk_params)
                kp_left_t1, st_2, err = cv.calcOpticalFlowPyrLK(img_right_gary_t0, img_left_gary_t1, kp_right_t1, None, None, **lk_params)
                kp_left_t0r, st_3, err = cv.calcOpticalFlowPyrLK(img_left_gary_t1, img_left_gary_t0, kp_left_t1, None, None, **lk_params)     

                d = abs(kp_left_t0 - kp_left_t0r).reshape(-1, 2).max(-1)
                print('d shape', d.shape)

                good = d < 1
                print('good:', good.sum())

                new_tracks = []
                print('len(self.tracks)', len(self.tracks))
                print('len(kp_left_t1.tracks)', kp_left_t1.shape)
                


                for tr, (x, y), good_flag in zip(self.tracks, kp_left_t1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                print( 'track count: %d' % len(self.tracks))
                # draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            # if self.frame_idx % self.detect_interval == 0:
            # if i == 0:
            if len(self.tracks) < 2000:
                mask = np.zeros_like(img_left_gary_t0)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                
                p = cv.goodFeaturesToTrack(img_left_gary_t0, mask = mask, **feature_params)
                # p = featureDetectionFast(img_left_gary_t0)
                # p = kp_to_pt(p)

                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])


            self.frame_idx += 1
            cv.imshow('lk_track', vis)

            # cv.namedWindow('left',cv.WINDOW_NORMAL)
            # cv.imshow('left', img_left_color_t0)
            # cv.resizeWindow('left', 360, 680)

            # cv.namedWindow('right',cv.WINDOW_NORMAL)
            # cv.imshow('right', img_right_color_t0)
            # cv.resizeWindow('right', 360, 680)

            ch = cv.waitKey(1)
            if ch == 27:
                break

    def monoTracking(self):

        i = 0
        for i in range(self.steps):
            t = self.t_init + i
            print('Time step: ', t)

            # _ret, frame = self.cam.read()
            path = self.img_prefix + '/' + '%06d' % t +'.png'
            print('path: ', path)
            frame = cv.imread(path)
            print('frame: ', frame.shape)
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st_0, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st_1, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                print( 'track count: %d' % len(self.tracks))
                # draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
            # if i == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                
                p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                # p = featureDetectionFast(frame_gray)
                # p = kp_to_pt(p)

                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])


            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('lk_track', vis)

            ch = cv.waitKey(1)
            if ch == 27:
                break


def main():
    import sys
    try:
        video_src = sys.argv[1]
        t_init = int(sys.argv[2])
        step = int(sys.argv[3])
    except:
        video_src = 0

    print(__doc__)
    App(video_src, t_init, step).stereoTracking()
    # App(video_src, t_init, step).monoTracking()

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()