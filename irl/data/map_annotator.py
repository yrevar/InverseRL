"""
Usage: map_annotator.py [OPTIONS]
"""
import numpy as np
import click, cv2, os, shutil, os.path as osp
import utils

def make_clean_dir(dirpath):
    if osp.exists(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)

class MapAnnotator(object):

    def __init__(self, map_fname, window='Map Annotator'):
        self._initialize(map_fname, window)
        self.launch_interface()

    def _initialize(self, map_fname, window):
        self.store_dir = osp.dirname(map_fname)
        self.store_fname = osp.splitext(osp.basename(map_fname))[0]
        self.map_img = cv2.imread(map_fname)
        self.window = window
        self.path = None
        self.path_list = {}
        self.path_idx = 0
        self.path_ready = False
        self.drawing = False
        self.pen_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        self.pen_thickness = 5
        self.state_wait_ms = 1  # refresh rate
        self.temp_state_wait_ms = 800  # exit state after delay
        self.help_menu_width = 400
        self.top_panel_height = 100
        self.relative_line_spacing = 1.5
        self.help_menu_fontscale = 0.8
        self.top_panel_fontscale = 1.0
        path_list = self.load_trajectories(self.get_trajectory_fname())
        if path_list is not None:
            self.path_list = path_list
            self.path_idx = len(path_list)
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.mouse_handler)

    def get_pen_color(self):
        return self.pen_color

    def _randomize_pen_color(self):
        self.pen_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

    def mouse_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.path = np.array([[x, y]], dtype=np.int32)
            self.path_list[self.path_idx] = (self.path, self.get_pen_color())

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                self.path = np.vstack((self.path, [x, y])).astype(np.int32)
                self.path_list[self.path_idx] = (self.path, self.get_pen_color())

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.path_ready = True
            self.path_idx += 1

    def get_trajectory_fname(self):
        return osp.join(self.store_dir, self.store_fname + "_" + "trajectories.npy")

    def load_trajectories(self, fname):
        if osp.exists(fname):
            paths = np.load(fname, allow_pickle=True)
            path_list = {}
            for idx, path in enumerate(paths):
                self._randomize_pen_color()
                path_list[idx] = (path, self.get_pen_color())
            return path_list
        else:
            return None

    def get_menu_info(self):
        return "Trajectories: {}\n" \
               "".format(self.path_idx)

    def render(self, window, img, wait_ms=0):
        cv2.imshow(window, img)
        return cv2.waitKey(wait_ms) & 0xFF

    def build_render_frame(self, img, fn_lst):
        img = img.copy()
        for fn in fn_lst:
            img = fn(img)
        return img

    def render_paths(self, img):
        if len(self.path_list) != 0:
            for k, v in self.path_list.items():
                path, pen_color = v
                img = cv2.polylines(img, [path], False, pen_color, self.pen_thickness)
                img = cv2.circle(img, tuple(path[0]), int(self.pen_thickness * 2.0), pen_color, -1)
                img = cv2.rectangle(img, (path[-1][0]-self.pen_thickness, path[-1][1]-self.pen_thickness),
                                    (path[-1][0]+self.pen_thickness, path[-1][1]+self.pen_thickness),
                                    pen_color, -1)
        return img

    def render_help_menu(self, img):
        start_x, start_y = 10, 20
        menu_img = np.ones((img.shape[0], self.help_menu_width, img.shape[2]), dtype=img.dtype) * np.array([[[150, 150, 150]]], dtype=img.dtype)
        # show pen color
        self.putMsg(menu_img, "Pen Color: ",  (start_x, start_y + 40),
                    h_align="left",
                    font_face=cv2.FONT_HERSHEY_TRIPLEX,
                    font_color=(0, 255, 0),
                    font_scale=self.help_menu_fontscale)

        cv2.rectangle(menu_img, (start_x + 170, start_y), (start_x + 250, start_y + 80), self.get_pen_color(), -1)
        end_y = start_y + 100
        helptext = "---Help--- \
                        \n'p': change pen color \
                        \n'<num>': pen thickness \
                        \n'c': clear last trajectory \
                        \n'x': clear all trajectories \
                        \n's': save trajectories \
                        \n'ESC': Exit"
        _ = self.putMsg(
            menu_img, helptext, (start_x, end_y),
            h_align="left",
            font_face=cv2.FONT_HERSHEY_TRIPLEX,
            font_color=(0, 255, 0),
            font_scale=self.help_menu_fontscale)
        img = np.hstack((img, menu_img))
        return img

    def render_update_overlay(self, img, shape=None, rel_position=None, msg=None, alpha=1.):
        if msg is not None:
            color = (200, 200, 200)
            overlay = img.copy()

            if shape is None:
                h, w = img.shape[0] // 4, img.shape[1] // 2
            else:
                h, w = shape

            if rel_position is None:
                x, y = int(img.shape[1] * (1. - 1/2)/2), int(img.shape[0] * (1. - 1/4)/2)
            else:
                x, y = self._rel_to_abs_position(overlay, rel_position)
            overlay[y:y+h, x:x+w, :] = list(color)
            self.putMsg(overlay, msg, (x+w//2, y+h//2), font_scale=0.8, h_align="center", font_color=(0, 0, 255))
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        return img

    def render_top_panel(self, img):
        h, w, c = img.shape
        top_panel =  np.ones((self.top_panel_height, w, c), dtype=img.dtype) * np.array([[[100, 100, 100]]], dtype=img.dtype)

        top_panel, end_x, end_y = self.putMsg(
            top_panel, self.get_menu_info(), (int((w-self.help_menu_width) * 0.5), int(self.top_panel_height * 0.5)),
            h_align="left",
            font_face=cv2.FONT_HERSHEY_TRIPLEX,
            font_color=(0, 255, 0),
            font_scale=self.top_panel_fontscale)

        img = np.vstack((top_panel, img))
        return img

    def putMsg(self, img, msg, position, font_face=cv2.FONT_HERSHEY_TRIPLEX,
               font_scale=1, font_color=(0, 0, 255), font_thickness=1,
               h_align="center", v_align="center"):

        H, W, _ = img.shape
        x_, y_ = position
        w_max, h_max = np.float("-inf"), np.float("-inf")
        line_wh = {}

        for i, line in enumerate(msg.split('\n')):
            (w, h), _ = cv2.getTextSize(line, font_face, font_scale, font_thickness)
            w_max = max(w, w_max)
            h_max = max(int(self.relative_line_spacing * h), h_max)
            line_wh[i] = w, h

        for i, line in enumerate(msg.split('\n')):

            w, h = line_wh[i]
            if h_align == "left":
                ln_x = x_
            elif h_align == "right":
                ln_x = x_ - w_max
            else:
                ln_x = x_ - w // 2
            if v_align == "top":
                ln_y = y_ + h
            elif v_align == "bottom":
                ln_y = y_ + h_max - h
            else:
                ln_y = y_ + h // 2
            cv2.putText(img, line, (ln_x, ln_y), font_face, font_scale, font_color, font_thickness)
            y_ += h_max
        return img, x_ + w_max, y_

    def _rel_to_abs_position(self, img, pos):

        H, W, _ = img.shape
        mW, mH = pos # Note: position is (x, y)
        if mH < 0:
            mH = 1. - abs(mH)
        if mW < 0:
            mW = 1. - abs(mW)
        return int(W * mW), int(H * mH)

    def putMsg_relative(self, img, msg, rel_position, font_face=cv2.FONT_HERSHEY_TRIPLEX, font_scale=1,
                   font_color=(0, 0, 255), font_thickness=1, h_align="center", v_align="center"):
        return self.putMsg(img, msg, self._rel_to_abs_position(img, rel_position), font_face, font_scale,
                    font_color, font_thickness, h_align, v_align)

    def display_current_state(self, window, event=None, msg=None, is_temp=False, override_wait_ms=None):
        render_img = self.build_render_frame(
            self.map_img,
            [
                lambda img: self.render_help_menu(img),
                lambda img: self.render_top_panel(img),
                lambda img: self.render_paths(img),
                lambda img: self.render_update_overlay(img, msg=msg),
            ]
        )
        if override_wait_ms:
            wait_time = override_wait_ms
        else:
            wait_time = self.temp_state_wait_ms if is_temp else self.state_wait_ms
        return render_img, self.render(window, render_img, wait_time)

    def display_temp_state(self, *args, **kwargs):
        kwargs["is_temp"] = True
        return self.display_current_state(*args, **kwargs)

    def launch_interface(self, debug=False):
        while True:
            render_img, event = self.display_current_state(self.window)
            if event != 255 and debug:
                print('Key pressed %d (0x%x), LSB: %d (%s)' % (
                    event, event, event % 256, repr(chr(event % 256)) if event % 256 < 128 else '?'))

            if ord("0") <= event <= ord("9"):
                label = event - ord("0")
                self.pen_thickness = 3 + 2 * label
            elif event == ord('p'):
                if debug: print("getting new pen color...")
                self._randomize_pen_color()
            elif event == ord('c'):
                if debug: print("clearing last trajectory...")
                if self.path_idx-1 in self.path_list:
                    del self.path_list[self.path_idx-1]
                    self.path_idx = self.path_idx-1
            elif event == ord('x'):
                if debug: print("clearing trajectories...")
                del self.path_list
                self.path_list = {}
                self.path_idx = 0
                _ = self.display_temp_state(self.window, msg="Cleared all trajectories at", override_wait_ms=800)
            elif event == ord('s'):
                if debug: print("saving trajectories...")
                paths = []
                for k, v in self.path_list.items():
                    # print(v[0].shape)
                    paths.append(v[0])
                np.save(self.get_trajectory_fname(), np.asarray(paths))
                _ = self.display_temp_state(self.window, msg="Saved {} trajectories at\n{}".format(
                    len(paths), self.get_trajectory_fname()), override_wait_ms=800)
            elif event == 27 or event in [ord('q'), ord('Q')]:
                if debug: print("Exiting...")
                _ = self.display_temp_state(
                    self.window, msg="Exiting...\nThank you for using this tool!", override_wait_ms=1000)
                cv2.destroyWindow(self.window)
                cv2.destroyAllWindows()
                break

@click.command()
@click.option('-i', '--input-map',  required=True, type=str, help="Input map file.")
def map_annotator(input_map):
    ma = MapAnnotator(input_map)

if __name__ == "__main__":
    map_annotator()
