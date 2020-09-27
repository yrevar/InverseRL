# Python imports
import glob, os.path as osp
import pandas as pd
from utils.VideoHelper import VideoHelper

# OpenCV
import cv2

class StanfordCampusDataset:

    def __init__(self, data_dir="../datasets/stanford_campus_dataset"):
        self.data_dir = data_dir
        self._parse_dataset(self.data_dir)

    @classmethod
    def name(cls):
        return cls.__name__

    def _parse_dataset(self, data_dir):
        self.scenes_dir_list = glob.glob(osp.join(data_dir, "annotations", "*"))
        self.scenes = [osp.basename(scene_dir) for scene_dir in self.scenes_dir_list]
        self.scene_to_videos_dict = {}
        for scene in self.scenes:
            self.scene_to_videos_dict[scene] = self.get_videos(scene)

    def get_scene_dir_list(self):
        return self.scenes_dir_list

    def get_scenes(self):
        return self.scenes

    def get_videos_dir_list(self, scene):
        return glob.glob(osp.join(self.data_dir, "videos", scene, "*"))

    def get_videos(self, scene):
        return [osp.basename(video_dir) for video_dir in self.get_videos_dir_list(scene)]

    def get_scene_video_map(self):
        return self.scene_to_videos_dict

    def get_annotation_file_path(self, scene, video):
        return osp.join(self.data_dir, "annotations", scene, video, "annotations.txt")

    def get_video_file_path(self, scene, video):
        return osp.join(self.data_dir, "videos", scene, video, "video.mov")

    def get_base_image_path(self, scene, video):
        return osp.join(self.data_dir, "annotations", scene, video, "reference.jpg")

    def get_annotations_df(self, scene, video):
        ann_file = self.get_annotation_file_path(scene, video)
        header_list = ["track_id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]
        df = pd.read_csv(ann_file, header=None, names=header_list, delimiter=" ")
        df["ycenter"] = ((df["ymin"] + df["ymax"]) / 2).astype(int)
        df["xcenter"] = ((df["xmin"] + df["xmax"]) / 2).astype(int)
        return df

    def get_track_ids(self, scene, video):
        return self.get_annotations_df(scene, video).track_id.unique()

    def get_track_df(self, scene, video, track_id):
        df = self.get_annotations_df(scene, video)
        return df[df.track_id == track_id]

    def get_track_list(self, scene, video, select_track_ids="all"):
        track_list = []
        assert select_track_ids in [None, "all"] or isinstance(select_track_ids, list)
        df = self.get_annotations_df(scene, video)
        for track_id, df_track in df.groupby("track_id"):
            if (not select_track_ids in [None, "all"]) \
                    and len(select_track_ids) != 0\
                    and isinstance(select_track_ids, list):
                if track_id in select_track_ids:
                    track_list.append(df_track[["xcenter", "ycenter"]].values)
            else:
                track_list.append(df_track[["xcenter", "ycenter"]].values)
        return track_list

    def show_video(self, scene, video,
                   fps=20, frame_start=None, frame_end=None,
                   select_track_ids="all", window = "test"):

        video_file = self.get_video_file_path(scene, video)
        ann_df = self.get_annotations_df(scene, video)

        if select_track_ids == "all":
            select_track_ids = ann_df.track_id.unique()

        video = VideoHelper(video_file, frame_start=frame_start, frame_end=frame_end)

        ann_df = ann_df.loc[ann_df['track_id'].isin(select_track_ids)]
        frame, df_frame_list = list(zip(*ann_df.groupby("frame")))

        while not video.is_finished():
            frame, frame_no = video.curr_frame()
            for i, df_frame in df_frame_list[frame_no].iterrows():
                cv2.rectangle(frame, (df_frame.xmin, df_frame.ymin),
                              (df_frame.xmax, df_frame.ymax),
                              (255, 0, 0), 2)
                cv2.putText(frame, "{}: {}".format(df_frame.track_id, df_frame.label),
                            (df_frame.xmin, df_frame.ymin),
                            cv2.FONT_HERSHEY_PLAIN,
                            1., (0, 255, 0), 2, cv2.LINE_AA)
                # cv2.putText(img, line, (ln_x, ln_y), font_face, font_scale, font_color, font_thickness)
            cv2.imshow(window, frame)
            if cv2.waitKey(int(1000 * 1 / int(fps))) & 0xFF == ord('q'):
                break
            video.seek_next()
        video.release()
        cv2.destroyWindow(window)
        cv2.destroyAllWindows()

    def create_video(self, scene, video, store_file="./video.avi",
                     fps=20, frame_start=None, frame_end=None,
                     select_track_ids="all", window = "test"):

        video_file = self.get_video_file_path(scene, video)
        ann_df = self.get_annotations_df(scene, video)

        if select_track_ids == "all":
            select_track_ids = ann_df.track_id.unique()

        video = VideoHelper(video_file, frame_start=frame_start, frame_end=frame_end)
        video_out = None

        ann_df = ann_df.loc[ann_df['track_id'].isin(select_track_ids)]
        frame, df_frame_list = list(zip(*ann_df.groupby("frame")))

        while not video.is_finished():
            frame, frame_no = video.curr_frame()
            print("Processing frame: {} / {}\r".format(frame_no + 1, video.length()), end="")
            if frame_no >= len(df_frame_list):
                print("frame {} missing in df".format(frame_no))
            else:
                for i, df_frame in df_frame_list[frame_no].iterrows():
                    cv2.rectangle(frame, (df_frame.xmin, df_frame.ymin),
                                  (df_frame.xmax, df_frame.ymax),
                                  (255, 0, 0), 2)
                    cv2.putText(frame, "{}: {}".format(df_frame.track_id, df_frame.label),
                                (df_frame.xmin, df_frame.ymin),
                                cv2.FONT_HERSHEY_PLAIN,
                                1., (0, 255, 0), 2, cv2.LINE_AA)
                    # cv2.putText(img, line, (ln_x, ln_y), font_face, font_scale, font_color, font_thickness)
                if video_out is None:
                    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                    video_out = cv2.VideoWriter(store_file, fourcc, fps, (frame.shape[1], frame.shape[0]))
                video_out.write(frame)
            video.seek_next()
        print("Finished writing video at {}".format(store_file))
        video.release()
        video_out.release()
        cv2.destroyWindow(window)
        cv2.destroyAllWindows()
