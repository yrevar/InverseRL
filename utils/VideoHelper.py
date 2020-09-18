import numpy as np
import cv2

class VideoHelper(object):

    def __init__(self, video_file, frame_start=0, frame_end=np.float('inf'), do_loop=True, preload_frames=500):

        self.preloaded = False
        self.preload_frames = preload_frames
        self._init_capture(video_file, frame_start, frame_end)
        self.do_loop = do_loop

    def restart_capture(self):
        self.curr_frame_idx = self.frame_start

    def curr_frame_no(self):
        return self.curr_frame_idx-self.frame_start

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

        if self.n_frames < self.preload_frames:
            self._preload()

    def _preload(self):

        self.frames = []
        self.preloaded = False

        for frame_no in range(self.frame_start, self.frame_end+1):
            self.frames.append(self.get_frame_by_idx(frame_no))

        self.preloaded = True

    def is_finished(self):
        return self.curr_frame_idx > self.frame_end and not self.do_loop

    def get_frame_by_idx(self, idx):

        if self.preloaded:
            idx -= self.frame_start
            return self.frames[idx]
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read(1)
            if ret == False:
                raise RuntimeError("Failed to read frame no. {} from file {}".format(idx, self.video_file))
            return frame

    def seek_next(self, skip=1):

        self.curr_frame_idx += skip

        if self.curr_frame_idx > self.frame_end:
            if self.do_loop:
                self.curr_frame_idx = self.frame_start
            else:
                self.curr_frame_idx = self.frame_end

    def seek_prev(self, skip=1):

        idx = max(self.frame_start, self.curr_frame_idx-skip)
        self.curr_frame_idx = idx

    def curr_frame(self):

        idx = self.curr_frame_idx

        if idx > self.frame_end:
            raise RuntimeError('No more frames! You should have known earlier by calling is_finished()')

        frame = self.get_frame_by_idx(idx)
        return frame, self.curr_frame_no()

    def next_frame(self, skip=1):

        self.seek_next(skip)
        return self.curr_frame()

    def prev_frame(self, skip=1):

        self.seek_prev(skip)
        return self.curr_frame()

    def __del__(self):
        if self.cap is not None:
            self.cap.release()
