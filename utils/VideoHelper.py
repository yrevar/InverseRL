import numpy as np
from collections import namedtuple
import cv2

CornerCrop = namedtuple("CornerCropParams", "x1 y1 x2 y2")

def read_video_frames(video_file, frame_start, frame_end, resize=None, corner_crop=None):
    video = VideoHelper(video_file, frame_start=frame_start, frame_end=frame_end, resize=resize, corner_crop=corner_crop)
    frames = []
    while not video.is_finished():
        frame, idx = video.curr_frame()
        # print(idx,)
        frames.append(frame)
        video.seek_next(1)
    return np.asarray(frames)[:,:,:,::-1]

class VideoHelper(object):

    def __init__(self, video_file, frame_start=0, frame_end=np.float('inf'), do_loop=False,
                 resize=None, corner_crop=None):
        self.resize = resize
        if frame_start is None:
            frame_start = 0
        if frame_end is None:
            frame_end = np.float('inf')
        self._init_capture(video_file, frame_start, frame_end)
        self.do_loop = do_loop
        self.ccrop = corner_crop
        self.finished = False

    def restart_capture(self):
        self.curr_frame_idx = self.frame_start

    def curr_frame_no(self):
        return self.curr_frame_idx #-self.frame_start

    def _init_capture(self, video_file, frame_start, frame_end):
        self.video_file = video_file
        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to initialize capture for file {}".format(video_file))

        self.n_orig_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_start = max(frame_start, 0)

        if frame_end < 0:
            self.frame_end = max(self.n_orig_frames+frame_end, self.frame_start)
        else:
            self.frame_end = min(self.n_orig_frames-1, frame_end)

        self.curr_frame_idx = self.frame_start
        self.n_frames = (self.frame_end - self.frame_start + 1)

    def is_finished(self):
        return self.finished and not self.do_loop

    def get_frame_by_idx(self, idx):
        # read video frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read(1)
        if ret == False:
            raise RuntimeError("Failed to read frame no. {} from file {}".format(idx, self.video_file))
        # crop image
        if self.ccrop is not None:
            frame = frame[self.ccrop.y1:self.ccrop.y2, self.ccrop.x1:self.ccrop.x2]
        # resize if needed
        if self.resize is not None:
            frame = cv2.resize(frame, self.resize)
        return frame

    def seek_next(self, skip=1):
        next_frame_idx = self.curr_frame_idx + skip
        if next_frame_idx > self.frame_end:
            if self.do_loop:
                self.curr_frame_idx = self.frame_start
                self.finished = False
            else:
                self.curr_frame_idx = self.frame_end
                self.finished = True
        else:
            self.curr_frame_idx = next_frame_idx
            self.finished = False

    def seek_prev(self, skip=1):
        idx = max(self.frame_start, self.curr_frame_idx-skip)
        self.curr_frame_idx = idx
        self.finished = False

    def seek(self, frame_idx):
        if frame_idx < 0 or frame_idx > self.frame_end:
            raise Exception("Can't seek to {}".format(frame_idx))
        self.finished = False
        self.curr_frame_idx = frame_idx

    def curr_frame(self):
        idx = self.curr_frame_idx
        #
        # if idx > self.frame_end:
        #     raise RuntimeError('No more frames! You should have known earlier by calling is_finished()')

        frame = self.get_frame_by_idx(idx)
        return frame, self.curr_frame_no()

    def next_frame(self, skip=1):
        self.seek_next(skip)
        return self.curr_frame()

    def prev_frame(self, skip=1):

        self.seek_prev(skip)
        return self.curr_frame()

    def length(self):
        return self.n_frames

    def release(self):
        if self.cap is not None:
            self.cap.release()

    def __del__(self):
        self.release()
