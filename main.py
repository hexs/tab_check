import json
import time
from pprint import pprint
from typing import Union, Tuple
import pygame
from func.constant import *
from datetime import datetime
import os
import cv2
import numpy as np


class Frame:
    def __init__(self, name: str, rect: Union[list, tuple], wh: Tuple[int, int] = (0, 0)):
        self.name = name
        self.rect = rect
        self.x, self.y, self.dx, self.dy = rect
        self.index__next__ = 0
        self.wh = wh
        self.img_w, self.img_h = wh

    def __str__(self):
        return (f'{UNDERLINE}{BOLD}{CYAN}{self.x}{ENDC}{UNDERLINE} {BOLD}{PINK}{self.y}{ENDC} -> '
                f'{UNDERLINE}{BOLD}{CYAN}{self.dx}{ENDC}{UNDERLINE} {BOLD}{PINK}{self.dy}{ENDC}')

    def __iter__(self):
        return self

    def __next__(self):
        self.index__next__ += 1
        if self.index__next__ == 1:
            return self.x
        elif self.index__next__ == 2:
            return self.y
        elif self.index__next__ == 3:
            return self.dx
        elif self.index__next__ == 4:
            return self.dy
        else:
            self.index__next__ = 0
            raise StopIteration

    def xywh(self):
        return self.x, self.y, self.dx, self.dy

    def pix_xywh(self, **kwargs):
        if kwargs.get('shape'):
            self.shape = kwargs.get('shape')
            self.img_w, self.img_h = self.shape
        x = round((self.x - self.dx / 2) * self.img_w)
        y = round((self.y - self.dy / 2) * self.img_h)
        w = round(self.dx * self.img_w)
        h = round(self.dy * self.img_h)
        return x, y, w, h

    def pix_xyxy(self, **kwargs):
        if kwargs.get('shape'):
            self.shape = kwargs.get('shape')
            self.img_w, self.img_h = self.shape
        x1 = round((self.x - self.dx / 2) * self.img_w)
        y1 = round((self.y - self.dy / 2) * self.img_h)
        x2 = round((self.x + self.dx / 2) * self.img_w)
        y2 = round((self.y + self.dy / 2) * self.img_h)
        return x1, y1, x2, y2

    def pix_x1(self, **kwargs):
        x1, y1, x2, y2 = self.pix_xyxy(**kwargs)
        return x1

    def pix_x2(self, **kwargs):
        x1, y1, x2, y2 = self.pix_xyxy(**kwargs)
        return x2

    def pix_y1(self, **kwargs):
        x1, y1, x2, y2 = self.pix_xyxy(**kwargs)
        return y1

    def pix_y2(self, **kwargs):
        x1, y1, x2, y2 = self.pix_xyxy(**kwargs)
        return y2

    def cv2_rectangle(self, img, **kwargs):
        x1 = int((self.x - self.dx / 2) * self.img_w)
        y1 = int((self.y - self.dy / 2) * self.img_h)
        x2 = int((self.x + self.dx / 2) * self.img_w)
        y2 = int((self.y + self.dy / 2) * self.img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    def save(self, img, path):
        path = f'data/tab/img_full'
        mkdir(path)
        namefile = datetime.now().strftime('%y%m%d-%H%M%S')

        cv2.imwrite(os.path.join(path, namefile + '.png'), img)

        string = f"{self.name}:{'ok'}\n"
        with open(os.path.join(path, namefile + '.txt'), 'a') as file:
            file.write(string)


class Model:

    def __init__(self, model_name: str, PCB_name: str, wh):
        self.model_name = model_name
        self.model = None
        self.PCB_name = PCB_name
        self.IMG_FULL_PATH = f'data/{self.PCB_name}/img_full'
        self.IMG_FRAME_PATH = f'data/{self.PCB_name}/img_frame'
        self.IMG_FRAME_LOG_PATH = f'data/{self.PCB_name}/img_frame_log'
        self.MODEL_PATH = f'data/{self.PCB_name}/models'

        self.wh = wh
        self.img_w, self.img_h = wh
        self.frames = {}
        self.class_names = []
        self.img_height = 180
        self.img_width = 180
        self.batch_size = 32
        self.epochs = 5

    def __str__(self):
        return f'{BLUE}Model({self.model_name}){ENDC}'

    def add_frame(self, name: str, rect: Union[list, tuple]):
        self.frames[name] = Frame(name, rect, self.wh)

    def corp_img(self):
        png_and_txt_list = os.listdir(self.IMG_FULL_PATH)
        png_and_txt_list = list(
            set(file.split('.')[0] for file in png_and_txt_list if file.endswith('.png')) &
            set(file.split('.')[0] for file in png_and_txt_list if file.endswith('.txt')))
        png_and_txt_list = sorted(png_and_txt_list, reverse=True)
        for i, file_name in enumerate(png_and_txt_list, start=1):
            # file_name is 0807 143021, ...
            frames_list = open(fr"{self.IMG_FULL_PATH}/{file_name}.txt").readlines()
            print(f'{i}/{len(png_and_txt_list)} {file_name}')
            for data_text in frames_list:
                frame_name, status = data_text.strip().split(':')  # _________________ ชื่อใน .txt และ class_names
                if frame_name not in self.frames.keys():  # __________________________ ชื่อใน .txt ไม่ตรง
                    continue
                img = cv2.imread(fr"{self.IMG_FULL_PATH}/{file_name}.png")
                pix_Y, pix_X, _ = img.shape
                x1, y1, x2, y2 = self.frames[frame_name].pix_xyxy(shape=(pix_X, pix_Y))
                print(img.shape)
                print(x1, y1, x2, y2)

                img_crop_namefile = f'{status} {frame_name} {file_name}.png'
                mkdir(fr"{self.IMG_FRAME_PATH}/{self.model_name}")
                mkdir(fr"{self.IMG_FRAME_PATH}/{self.model_name}/{status}")
                mkdir(fr"{self.IMG_FRAME_LOG_PATH}/{self.model_name}")

                img_crop = img[y1:y2, x1:x2]
                cv2.imwrite(fr"{self.IMG_FRAME_LOG_PATH}/{self.model_name}/{img_crop_namefile}", img_crop)
                shift = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
                # shift = [-2, 0]
                brightness = [230, 242, 255, 267, 280]
                contrast = [114, 120, 127, 133, 140]
                # brightness = [242, 255, 267]
                # contrast = [120, 127, 133]
                for shift_y in shift:
                    for shift_x in shift:
                        img_crop = img[y1 + shift_y:y2 + shift_y, x1 + shift_x:x2 + shift_x]

                        # add contran blige
                        for b in brightness:
                            for c in contrast:
                                img_crop_BC = img_crop.copy()
                                img_crop_BC = controller(img_crop_BC, b, c)

                                img_crop_namefile = f'{file_name} {frame_name} {status} {shift_y} {shift_x} {b} {c}.png'
                                cv2.imwrite(fr"{self.IMG_FRAME_PATH}/{self.model_name}/{status}/{img_crop_namefile}",
                                            img_crop_BC)
        print()

    def load_model(self):
        from keras import models
        # path = os.path.join(f'{self.MODEL_PATH}/model_name.h5')
        path = os.path.join(f'{self.MODEL_PATH}/{self.model_name}.h5')
        print(path)
        self.model = models.load_model(path)
        with open(os.path.join(self.MODEL_PATH, self.model_name + '.json')) as f:
            string = f.read()
        self.class_names = json.loads(string)

    def _predict_(self, img):
        for frame_name, frame in self.frames.items():
            x1, y1, x2, y2 = frame.pix_xyxy()
            img_crop = img[y1:y2, x1:x2]
            img_resized = cv2.resize(img_crop, (180, 180))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_array4 = np.expand_dims(img_rgb, axis=0)

            predictions = self.model.predict_on_batch(img_array4)
            exp_x = [2.7 ** x for x in predictions[0]]
            percent_score_list = [round(x * 100 / sum(exp_x)) for x in exp_x]
            highest_score_index = np.argmax(predictions[0])  # 3
            highest_score_name = self.class_names[highest_score_index]
            highest_score_percent = percent_score_list[highest_score_index]
            return highest_score_name, highest_score_percent

    def fit_model(self):
        import shutil
        import tensorflow as tf
        from keras import layers, models
        from keras.models import Sequential
        import pathlib
        import matplotlib.pyplot as plt

        data_dir = pathlib.Path(rf'{self.IMG_FRAME_PATH}/{self.model_name}')
        image_count = len(list(data_dir.glob('*/*.png')))
        print(f'image_count = {image_count}')

        print('data_dir is ', data_dir)
        print(555, data_dir)
        train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="both",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.class_names = train_ds.class_names
        print('train_ds.class_names =', self.class_names)
        with open(fr'{self.MODEL_PATH}/{self.model_name}.json', 'w') as file:
            file.write(json.dumps(self.class_names, indent=4))

        # Visualize the data
        plt.figure(figsize=(20, 10))
        for images, labels in train_ds.take(1):
            for i in range(32):
                ax = plt.subplot(4, 8, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.class_names[labels[i]])
                plt.axis("off")
        plt.savefig(f'{self.MODEL_PATH}/{self.model_name}.png')

        for image_batch, labels_batch in train_ds:
            print(image_batch.shape)
            print(labels_batch.shape)
            break

        # Configure the dataset for performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Standardize the data
        normalization_layer = layers.Rescaling(1. / 255)

        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]
        # Notice the pixel values are now in `[0,1]`.
        print(np.min(first_image), np.max(first_image))

        # Create the model
        num_classes = len(self.class_names)

        self.model = Sequential([
            layers.Rescaling(1. / 255, input_shape=(self.img_height, self.img_width, 3)),
            layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.summary()

        history = self.model.fit(train_ds, validation_data=val_ds, epochs=self.epochs)

        # Visualize training results
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        loss = [1 if a > 1 else a for a in loss]
        val_loss = [1 if a > 1 else a for a in val_loss]

        epochs_range = range(self.epochs)
        plt.figure(figsize=(10, 8))
        plt.plot(epochs_range, val_acc, label='Validation Accuracy', c=(0, 0.8, 0.5))
        plt.plot(epochs_range, acc, label='Training Accuracy', ls='--', c=(0, 0, 1))
        plt.plot(epochs_range, val_loss, label='Validation Loss', c=(1, 0.5, 0.1))
        plt.plot(epochs_range, loss, label='Training Loss', c='r', ls='--')
        plt.legend(loc='right')
        plt.title(self.model_name)

        # plt.show()
        plt.savefig(fr'{self.MODEL_PATH}/{self.model_name}_graf.png')
        self.model.save(os.path.join(self.MODEL_PATH, f'{self.model_name}.h5'))
        # delete IMG_FRAME_PATH
        # shutil.rmtree(fr"{self.IMG_FRAME_PATH}/{self.model_name}")


class Models:
    def __init__(self, PCB_name, wh):
        self.models = {}
        self.PCB_name = PCB_name
        self.wh = wh
        self.img_w, self.img_h = wh
        self.class_names_set = []

    def add_model(self, name):
        self.models[name] = Model(name, self.PCB_name, self.wh)

    def load(self):
        with open(f'data/{self.PCB_name}/models.json', ) as f:
            d = json.loads(f.read())
            pprint(d)
            print()
            for model_name, v in d.items():
                self.class_names_set = v['class_names_set']
                frames = v['frames']
                self.add_model(model_name)
                for frame_name, vv in frames.items():
                    xywh = vv['xywh']
                    self.models[model_name].add_frame(frame_name, xywh)

    def load_model(self):
        for k, model in self.models.items():
            print(f'{PINK}load model {k}{ENDC}')
            model.load_model()

    def predict(self, img: np):
        for k, m in self.models.items():
            img2 = img.copy()
            print(f'{PINK}predict {k}{ENDC}')
            r = m._predict_(img2)
            print(r)

    def to_dict(self):
        d = {}
        for k, model in self.models.items():
            d[k] = {}
            d[k]['frames'] = {}
            for kk, frame in model.frames.items():
                d[k]['frames'][kk] = {'xywh': frame.xywh()}
            d[k]['class_names'] = model.class_names
        pprint(d)
        with open(f'data/{self.PCB_name}/models.json', 'w') as f:
            f.write(json.dumps(d, indent=4))

    def create_model(self):
        IMG_FULL_PATH = f'data/{self.PCB_name}/img_full'
        IMG_FRAME_PATH = f'data/{self.PCB_name}/img_frame'
        IMG_FRAME_LOG_PATH = f'data/{self.PCB_name}/img_frame_log'
        MODEL_PATH = f'data/{self.PCB_name}/models'
        mkdir(IMG_FULL_PATH)
        mkdir(IMG_FRAME_PATH)
        mkdir(IMG_FRAME_LOG_PATH)
        mkdir(MODEL_PATH)
        for name, model in m.models.items():
            model.corp_img()
            model.fit_model()


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    w, h = cap.get(3), cap.get(4)
    print(w, h)
    m = Models('tab', (w, h))

    m.load()
    # m.add_model('m1')
    # m.models['m1'].add_frame('tap1', (0.08, 0.55, 0.13, 0.18))
    # m.add_model('m2')
    # m.models['m2'].add_frame('tap2', (0.86, 0.64, 0.2, 0.44))

    m.create_model()
    # m.load_model()

    t2 = datetime.now()
    while True:
        t1 = t2
        t2 = datetime.now()
        fps = round(1 / max(0.001, (t2 - t1).total_seconds()))
        _, img = cap.read()
        if _:
            for k, v in m.models.items():
                for kk, vv in v.frames.items():
                    vv.cv2_rectangle(img)
            cv2.rectangle(img, (0, 0), (50, 18), (255, 255, 255), -1)
            cv2.putText(img, f'{fps}', (5, 15), 1, 1, (255, 0, 0), 1)
            cv2.imshow('img', img)
            key = cv2.waitKey(1)

            if key == ord(' '):
                print('save')
                mkdir('img_full')
                for k, v in m.models.items():
                    for kk, vv in v.frames.items():
                        vv.save(img, rf'img_full')

            if key == ord('p'):
                dt1 = datetime.now()
                m.predict(img)
                dt2 = datetime.now()
                print('predict time =', (dt2 - dt1).total_seconds())
                # predict time in computer = 0.02
                print()
        else:
            cap = cv2.VideoCapture(0)
            print('sleep(2)')
            time.sleep(2)
