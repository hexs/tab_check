import json
import random
import time
from datetime import datetime
import cv2
import numpy as np
from keras import models
from func.about_image import putTextRectlist
import sys

BLACK = '\033[90m'
FAIL = '\033[91m'
GREEN = '\033[92m'
WARNING = '\033[93m'
BLUE = '\033[94m'
PINK = '\033[95m'
CYAN = '\033[96m'
ENDC = '\033[0m'
BOLD = '\033[1m'
ITALICIZED = '\033[3m'
UNDERLINE = '\033[4m'


class Dtime():
    def __init__(self):
        self.t1 = {}  # {t1:datetime}
        self.s_list = {}  # {t1:[]}

    def start(self, t):
        self.t1[t] = datetime.now()

    def stop(self, t):
        t2 = datetime.now()
        dt_seconds = (t2 - self.t1[t]).total_seconds()
        if t not in self.s_list.keys():
            self.s_list[t] = []
        self.s_list[t].append(dt_seconds)

    def reset(self):
        for i in range(len(self.s_list)):
            self.s_list[i] = []

    def show(self):
        try:
            for t, v in self.s_list.items():
                dtime = self.s_list[t][-1]
                lenlist = len(self.s_list[t])
                mean = f'{sum(self.s_list[t]) / lenlist:.3f}'
                tmin = f'{min(self.s_list[t]):.3f}'
                tmax = f'{max(self.s_list[t]):.3f}'

                print(f'{dtime:.3f}s / min = {tmin}s / max = {tmax}s / mean = {mean}s  <--{t} {lenlist}')
            print()
        except:
            pass


def drawline(img, pt1, pt2, color, thickness=1, gap=10):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    e = pts[0]
    i = 0
    for p in pts:
        s = e
        e = p
        if i % 2 == 1:
            cv2.line(img, s, e, color, thickness)
        i += 1


def drawpoly(img, pts, color, thickness=1):
    for i in range(len(pts)):
        s = pts[i - 1]
        e = pts[i]
        drawline(img, s, e, color, thickness)


def drawrect(img, pt1, pt2, color, thickness=1):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness)


class Frame:
    def __init__(self, name, x, y, dx, dy, model_used, res_show, pcb_frame_name=None):
        self.name = name
        if pcb_frame_name:
            self.pcb_frame_name = pcb_frame_name
        else:
            self.pcb_frame_name = name
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.model_used = model_used
        self.res_show = res_show
        self.x1 = x - dx / 2
        self.y1 = y - dy / 2
        self.x2 = x + dx / 2
        self.y2 = y + dy / 2
        self.img = None
        self.debug_res_name = '-'
        self.reset_result()
        self.K_color = {
            'ok': (0, 255, 0),
            'nopart': (0, 0, 255),
            'wrongpart': (0, 150, 255),
            'wrongpolarity': (150, 0, 255)
        }

    def __str__(self):
        return (f'{PINK}Frame '
                f'{GREEN}{self.name}{ENDC}')

    def reset_result(self):
        self.color_frame = (0, 255, 255)
        self.color_frame_thickness = 5
        self.color_text = (255, 255, 255)
        self.font_size = 2.5
        self.predictions_score_list = None  # [ -6.520611   8.118368 -21.86103   22.21528 ]
        self.percent_score_list = None  # [3.3125e-11 7.5472e-05 7.2094e-18 9.9999e+01]
        self.highest_score_number = None  # ตำแหน่งไหน # 3
        self.highest_score_percent = None
        self.highest_score_name = None

    def resShow(self):
        for key, values in self.res_show.items():
            if self.highest_score_name in values:
                return key
        return self.highest_score_name


class Model:
    def __init__(self, name, status_list):
        self.name = name
        self.status_list = status_list
        self.model = None

    def __str__(self):
        return (f'{BLUE}Model '
                f'{GREEN}{self.name}{ENDC}')

    def load_model(self, modelname):
        try:
            self.model = models.load_model(fr'data/{modelname}/model/{self.name}.h5')
        except Exception as e:
            print(f'{WARNING}function "load_model" error.\n'
                  f'file error data/{modelname}/model/{self.name}.h5{ENDC}\n'
                  f'{str(e)}')
            # sys.exit()
        try:

            status_list = json.loads(open(fr'data/{modelname}/model/{self.name}.json').read())
            if status_list != self.status_list:
                print(f'{WARNING}status_list model != self.status_list')
                print(f'status_list from model = {status_list}')
                print(f'self.status_list       = {self.status_list}{ENDC}')

        except Exception as e:
            print(f'{WARNING}function "load_model" error.\n'
                  f'file error data/{modelname}/model/{self.name}.json{ENDC}'
                  f'{str(e)}')
            # sys.exit()


class Mark:
    def __init__(self, name, x, y, dx, dy, k):
        self.name = name
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.k = k

        self.x1 = x - dx / 2
        self.y1 = y - dy / 2
        self.x2 = x + dx / 2
        self.y2 = y + dy / 2

    def __str__(self):
        return (f'{CYAN}Mark '
                f'{GREEN}{self.name}{ENDC}')

    def rec_around(self, h, w):
        x1px = int(self.x1 * w)
        y1px = int(self.y1 * h)
        x2px = int(self.x2 * w)
        y2px = int(self.y2 * h)
        kx = int((x2px - x1px) * self.k)
        ky = int((y2px - y1px) * self.k)
        return (x1px - kx, y1px - ky), (x2px + kx, y2px + ky)

    def rec_mark(self, h, w):
        x1px = int(self.x1 * w)
        y1px = int(self.y1 * h)
        x2px = int(self.x2 * w)
        y2px = int(self.y2 * h)
        return (x1px, y1px), (x2px, y2px)

    def xpx(self, h, w):
        return int(self.x * w)

    def ypx(self, h, w):
        return int(self.y * h)

    def xypx(self, h, w):
        return self.xpx(h, w), self.ypx(h, w)


class Frames:
    def __init__(self, name):
        self.name = name
        data_all = json.loads(open(f'data/{name}/frames pos.json').read())
        self.frames = {}
        self.models = {}
        self.marks = {}

        for name, v in data_all['frames'].items():
            x = v['x']
            y = v['y']
            dx = v['dx']
            dy = v['dy']
            model_used = v['model_used']
            res_show = v['res_show']
            pcb_frame_name = v.get('pcb_frame_name')
            self.frames[name] = Frame(name, x, y, dx, dy, model_used, res_show, pcb_frame_name)
        for name, v in data_all['models'].items():
            status_list = sorted(v['status_list'])
            self.models[name] = Model(name, status_list)
        if data_all.get('marks'):
            for name, v in data_all['marks'].items():
                x = v['x']
                y = v['y']
                dx = v['dx']
                dy = v['dy']
                k = v['k']
                self.marks[name] = Mark(name, x, y, dx, dy, k)

        self.len = len(self.frames)
        self.color_frame = (255, 255, 255)

    def __str__(self):
        return f'{PINK}{BOLD}        ╔ {ENDC}{len(self.frames)} frame is {GREEN}{", ".join(self.frames.keys())}{ENDC}\n' \
               f'{PINK}{BOLD} Frames ╣ {ENDC}{len(self.models)} model is {GREEN}{", ".join(self.models.keys())}{ENDC}\n' \
               f'{PINK}{BOLD}        ╚ {ENDC}{len(self.marks)}  mark  is {GREEN}{", ".join(self.marks.keys())}{ENDC}\n'

    def crop_img(self, img_for_crop):
        h, w, _ = img_for_crop.shape
        for name, frame in self.frames.items():
            x1 = int(frame.x1 * w)
            y1 = int(frame.y1 * h)
            x2 = int(frame.x2 * w)
            y2 = int(frame.y2 * h)
            frame.img = img_for_crop[y1:y2, x1:x2]
            frame.img = cv2.resize(frame.img, (180, 180))

    def draw_blink(self, img, name):
        h, w, _ = img.shape
        print('name', name)
        for name, frame in self.frames.items():
            start_point = int(frame.x1 * w - 10), int(frame.y1 * h - 10)
            end_point = int(frame.x2 * w + 10), int(frame.y2 * h + 10)
            drawrect(img, start_point, end_point, (0, 255, 255), 12)

    def draw_frame(self, img, textdata=None):
        h, w, _ = img.shape
        for name, mark in self.marks.items():
            cv2.rectangle(img, *mark.rec_mark(h, w), (0, 255, 255), 2)
            cv2.rectangle(img, *mark.rec_around(h, w), (0, 255, 255), 2)

        for name, frame in self.frames.items():
            start_point = int(frame.x1 * w), int(frame.y1 * h)
            end_point = int(frame.x2 * w), int(frame.y2 * h)
            img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), frame.color_frame_thickness + 3)
            drawrect(img, start_point, end_point, frame.color_frame, frame.color_frame_thickness + 1)

        for name, frame in self.frames.items():
            start_point = int(frame.x1 * w), int(frame.y1 * h)
            end_point = int(frame.x2 * w), int(frame.y2 * h)
            if textdata:
                Abbreviation = {
                    'ok': 'ok',
                    'nopart': 'nopart',
                    'wrongpart': 'w.part',
                    'wrongpolarity': 'w.pola'
                }
                TextRectlist = []
                if textdata.get('Show name'):
                    d = f'{frame.name}'
                    if textdata.get('Change color name'):
                        d = (d, frame.color_text)
                    TextRectlist.append(d)
                if textdata.get('mode_debug'):
                    d = f'{frame.debug_res_name}'
                    d = Abbreviation[d] if (d in Abbreviation.keys()) else d
                    if textdata.get('Change color mode_debug'):
                        d = (d, frame.color_text)
                    TextRectlist.append(d)
                if textdata.get('Show results from predictions'):
                    d = f'{frame.highest_score_name}'
                    d = Abbreviation[d] if (d in Abbreviation.keys()) else d
                    if textdata.get('Change color results from predictions'):
                        d = (d, frame.color_text)
                    TextRectlist.append(d)
                if textdata.get('Show %results from predictions'):
                    d = f'{frame.highest_score_percent}'
                    if textdata.get('Change color %results from predictions'):
                        d = (d, frame.color_text)
                    TextRectlist.append(d)
                if textdata.get('Show list class name'):
                    d = f'{frame.percent_score_list}'
                    if textdata.get('Change color list class name'):
                        d = (d, frame.color_text)
                    TextRectlist.append(d)
                putTextRectlist(img, TextRectlist, 22, start_point, frame.font_size, 2, font=1, offset=1)
        return img

    def save_mark(self, img):
        h, w, _ = img.shape
        for name, mark in self.marks.items():
            x = mark.x
            y = mark.y
            dx = mark.dx
            dy = mark.dy
            x1 = int((x - dx / 2) * w)
            y1 = int((y - dy / 2) * h)
            x2 = int((x + dx / 2) * w)
            y2 = int((y + dy / 2) * h)
            print(f'{GREEN}save "data/{self.name}/{name}.png"{ENDC}')
            cv2.imwrite(f'data/{self.name}/{name}.png', img[y1:y2, x1:x2])
            cv2.imshow('img', img[y1:y2, x1:x2])


def predict(frame, Frames):
    model = Frames.models[frame.model_used]
    img_array = frame.img[np.newaxis, :]
    predictions = model.model.predict_on_batch(img_array)

    frame.predictions_score_list = predictions[0]  # [ -6.520611   8.118368 -21.86103   22.21528 ]
    exp_x = [1.2 ** x for x in frame.predictions_score_list]
    frame.percent_score_list = [round(x * 100 / sum(exp_x)) for x in exp_x]
    frame.highest_score_number = np.argmax(frame.predictions_score_list)  # 3

    frame.highest_score_name = model.status_list[frame.highest_score_number]
    frame.highest_score_percent = frame.percent_score_list[frame.highest_score_number]

    # if frame.highest_score_name:
    #     frame.color_frame = frame.color_text = frame.K_color[frame.highest_score_name]
    if frame.highest_score_name in frame.res_show['OK']:
        frame.color_frame = frame.color_text = (0, 255, 0)
    else:
        frame.color_frame = frame.color_text = (0, 0, 255)

    return frame.percent_score_list, predictions[0]


if __name__ == '__main__':
    framesmodel = Frames(rf"data\{'d07'}\frames pos.json")
    print(framesmodel)
    # print(frame.model_used)

    # img = np.full((1080, 1920, 3), (10, 10, 10), np.uint8)
    # frames = Frames(json.loads(open(r"data\D07\frames pos.json").read()))
    # frames.add_mark(json.loads(open(rf"data\D07\mark pos.json").read()))
    # print(frames)
    # print(frames.frames[0])
    # print(frames.frames[0].x)
    #
    # frames.draw_frame(img)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
