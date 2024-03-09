import json
import time
from pprint import pprint
from typing import Union, Tuple
import pygame
from func.constant import *
from datetime import datetime
import os
import cv2


class Frame:
    def __init__(self, name: str, rect: Union[list, tuple], shape: Tuple[int, int] = (0, 0)):
        self.name = name
        self.rect = rect
        self.x, self.y, self.dx, self.dy = rect
        self.index__next__ = 0
        self.shape = shape

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
        x = round((self.x - self.dx / 2) * self.shape[0])
        y = round((self.y - self.dy / 2) * self.shape[1])
        w = round(self.dx * self.shape[0])
        h = round(self.dy * self.shape[1])
        return x, y, w, h

    def pix_xyxy(self, **kwargs):
        if kwargs.get('shape'):
            self.shape = kwargs.get('shape')
        x1 = round((self.x - self.dx / 2) * self.shape[0])
        y1 = round((self.y - self.dy / 2) * self.shape[1])
        x2 = round((self.x + self.dx / 2) * self.shape[0])
        y2 = round((self.y + self.dy / 2) * self.shape[1])
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
        if kwargs.get('shape'):
            self.shape = kwargs.get('shape')
        x1 = int((self.x - self.dx / 2) * self.shape[0])
        y1 = int((self.y - self.dy / 2) * self.shape[1])
        x2 = int((self.x + self.dx / 2) * self.shape[0])
        y2 = int((self.y + self.dy / 2) * self.shape[1])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    def save(self, img, path):
        path = f'img_full'
        mkdir(path)
        namefile = datetime.now().strftime('%y%m%d-%H%M%S')

        cv2.imwrite(os.path.join(path, namefile + '.png'), img)

        string = f"{self.name}:{'ng'}\n"
        with open(os.path.join(path, namefile + '.txt'), 'a') as file:
            file.write(string)


class Model:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.frames = {}
        self.class_name = []
        self.img_height = 180
        self.img_width = 180
        self.batch_size = 32
        self.epochs = 8

    def add_frame(self, name: str, rect: Union[list, tuple]):
        self.frames[name] = Frame(name, rect)

    def corp_img(self, PCB_name):
        IMG_FULL_PATH = f'data/{PCB_name}/img_full'
        IMG_FRAME_PATH = f'data/{PCB_name}/img_frame'
        IMG_FRAME_LOG_PATH = f'data/{PCB_name}/img_frame_log'
        MODEL_PATH = f'data/{PCB_name}/models'

        png_and_txt_list = os.listdir(IMG_FULL_PATH)
        png_and_txt_list = list(
            set(file.split('.')[0] for file in png_and_txt_list if file.endswith('.png')) &
            set(file.split('.')[0] for file in png_and_txt_list if file.endswith('.txt')))
        png_and_txt_list = sorted(png_and_txt_list, reverse=True)
        for i, file_name in enumerate(png_and_txt_list, start=1):
            # file_name is 0807 143021, ...
            frames_list = open(fr"{IMG_FULL_PATH}/{file_name}.txt").readlines()
            print(f'{i}/{len(png_and_txt_list)} {file_name}')
            for data_text in frames_list:
                frame_name, status = data_text.strip().split(':')  # _________________ ชื่อใน .txt และ class_name
                if frame_name not in self.frames.keys():  # __________________________ ชื่อใน .txt ไม่ตรง
                    continue
                img = cv2.imread(fr"{IMG_FULL_PATH}/{file_name}.png")
                pix_Y, pix_X, _ = img.shape
                x1, y1, x2, y2 = self.frames[frame_name].pix_xyxy(shape=(pix_X, pix_Y))

                img_crop_namefile = f'{status} {frame_name} {file_name}.png'
                mkdir(fr"{IMG_FRAME_PATH}/{self.model_name}")
                mkdir(fr"{IMG_FRAME_PATH}/{self.model_name}/{status}")
                mkdir(fr"{IMG_FRAME_LOG_PATH}/{self.model_name}")

                img_crop = img[y1:y2, x1:x2]
                cv2.imwrite(fr"{IMG_FRAME_LOG_PATH}/{self.model_name}/{img_crop_namefile}", img_crop)

                for shift_y in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
                    for shift_x in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
                        img_crop = img[y1 + shift_y:y2 + shift_y, x1 + shift_x:x2 + shift_x]

                        # add contran blige
                        brightness = [230, 242, 255, 267, 280]
                        contrast = [114, 120, 127, 133, 140]
                        for b in brightness:
                            for c in contrast:
                                img_crop_BC = img_crop.copy()
                                img_crop_BC = controller(img_crop_BC, b, c)

                                img_crop_namefile = f'{file_name} {frame_name} {status} {shift_y} {shift_x} {b} {c}.png'
                                cv2.imwrite(fr"{IMG_FRAME_PATH}/{self.model_name}/{status}/{img_crop_namefile}",
                                            img_crop_BC)
        print()

    def fit_model(self, PCB_name):
        import shutil
        import tensorflow as tf
        from keras import layers, models
        from keras.models import Sequential
        import pathlib
        import matplotlib.pyplot as plt

        IMG_FULL_PATH = f'data/{PCB_name}/img_full'
        IMG_FRAME_PATH = f'data/{PCB_name}/img_frame'
        IMG_FRAME_LOG_PATH = f'data/{PCB_name}/img_frame_log'
        MODEL_PATH = f'data/{PCB_name}/models'

        data_dir = pathlib.Path(rf'{IMG_FRAME_PATH}/{self.model_name}')
        image_count = len(list(data_dir.glob('*/*.png')))
        print(f'image_count = {image_count}')

        print('data_dir is ', data_dir)
        print(555,data_dir)
        train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="both",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        class_names = train_ds.class_names
        print('class_names =', class_names)
        with open(fr'{MODEL_PATH}/{self.model_name}.json', 'w') as file:
            file.write(json.dumps(class_names, indent=4))

        # Visualize the data
        plt.figure(figsize=(20, 10))
        for images, labels in train_ds.take(1):
            for i in range(32):
                ax = plt.subplot(4, 8, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
        plt.savefig(f'{MODEL_PATH}/{self.model_name}.png')

        for image_batch, labels_batch in train_ds:
            print(image_batch.shape)
            print(labels_batch.shape)
            break

        # Configure the dataset for performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Standardize the data
        # normalization_layer = layers.Rescaling(1. / 255)

        # normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        # image_batch, labels_batch = next(iter(normalized_ds))
        # first_image = image_batch[0]
        # # Notice the pixel values are now in `[0,1]`.
        # print(np.min(first_image), np.max(first_image))

        # Create the model
        num_classes = len(class_names)

        model = Sequential([
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
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.summary()
        history = model.fit(train_ds, validation_data=val_ds, epochs=self.epochs)

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
        plt.savefig(fr'{MODEL_PATH}/{self.model_name}_graf.png')
        model.save(os.path.join(MODEL_PATH, f'{self.model_name}.h5'))
        # delete IMG_FRAME_PATH
        shutil.rmtree(fr"{IMG_FRAME_PATH}/{self.model_name}")



class Models:
    def __init__(self, PCB_name):
        self.models = {}
        self.PCB_name = PCB_name

    def add_model(self, name):
        self.models[name] = Model(name)

    def load(self):
        with open(f'data/{self.PCB_name}/models.json', ) as f:
            d = json.loads(f.read())
            pprint(d)
            print()
            for model_name, v in d.items():
                class_name = v['class_name']
                frames = v['frames']
                self.add_model(model_name)
                for frame_name, vv in frames.items():
                    xywh = vv['xywh']
                    self.models[model_name].add_frame(frame_name, xywh)
                self.class_name = class_name

    def to_dict(self):
        d = {}
        for k, model in self.models.items():
            d[k] = {}
            d[k]['frames'] = {}
            for kk, frame in model.frames.items():
                d[k]['frames'][kk] = {'xywh': frame.xywh()}
            d[k]['class_name'] = model.class_name
        pprint(d)
        with open(f'data/{self.PCB_name}/models.json', 'w') as f:
            f.write(json.dumps(d, indent=4))

    def create_model(self):
        IMG_FULL_PATH = f'data/{self.PCB_name}/img_full'
        IMG_FRAME_PATH = f'data/{self.PCB_name}/img_full'
        IMG_FRAME_LOG_PATH = f'data/{self.PCB_name}/img_frame_log'
        MODEL_PATH = f'data/{self.PCB_name}/models'
        mkdir(IMG_FULL_PATH)
        mkdir(IMG_FRAME_PATH)
        mkdir(IMG_FRAME_LOG_PATH)
        mkdir(MODEL_PATH)
        for name, model in m.models.items():
            model.corp_img(self.PCB_name)
            model.fit_model(self.PCB_name)


if __name__ == '__main__':
    # cap = cv2.VideoCapture(0)
    # w, h = cap.get(3), cap.get(4)
    w, h = 640.0, 480.0
    m = Models('tab')
    m.load()
    # m.add_model('m1')
    # m.models['m1'].add_frame('tap1', (0.08, 0.55, 0.13, 0.18))
    # m.add_model('m2')
    # m.models['m2'].add_frame('tap2', (0.86, 0.64, 0.2, 0.44))

    m.create_model()
    # while True:
    #     _, img = cap.read()
    #     for k, v in m.models.items():
    #         for kk, vv in v.frame.items():
    #             vv.rect.cv2_rectangle(img)
    #     cv2.imshow('img', img)
    #     key = cv2.waitKey(1)
    #
    #     if key == ord(' '):
    #         print('save')
    #         mkdir('img_full')
    #         for k, v in m.models.items():
    #             for kk, vv in v.frame.items():
    #                 vv.save(img, rf'img_full')
