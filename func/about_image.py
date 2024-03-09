import json
from datetime import datetime
import numpy as np
import cv2
import math
from scipy import ndimage


def putTextRect(img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255), colorR=(127, 127, 127),
                font=cv2.FONT_HERSHEY_PLAIN, offset=10, border=None, colorB=(0, 255, 0), cen=False):
    """
    Creates Text with Rectangle Background
    :param img: Image to put text rect on
    :param text: Text inside the rect
    :param pos: Starting position of the rect x1,y1
    :param scale: Scale of the text
    :param thickness: Thickness of the text
    :param colorT: Color of the Text
    :param colorR: Color of the Rectangle
    :param font: Font used. Must be cv2.FONT....
    :param offset: Clearance around the text
    :param border: Outline around the rect
    :param colorB: Color of the outline
    :return: image, rect (x1,y1,x2,y2)
    """
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    if cen:
        ox -= w // 2
        oy -= h // 2
    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset
    if offset:
        cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), colorB, border)
    cv2.putText(img, text, (ox, oy), font, scale, colorT, thickness)

    return img


def putTextRectlist(img, texts, step, pos, scale=3, thickness=3, colorR=(34, 31, 30), font=cv2.FONT_HERSHEY_PLAIN,
                    offset=10, border=None, colorB=(0, 255, 0), cen=False):
    # พื้นหลัง
    for i, text in enumerate(texts):
        if text not in ['None']:
            if type(text) is tuple:
                txt = text[0]
            else:
                txt = text

            ox, oy = pos
            oy = int(oy + i * step)
            (w, h), _ = cv2.getTextSize(txt, font, scale, thickness)
            if cen:
                ox -= w // 2
                oy -= h // 2
            x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset
            if offset:
                cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
            if border is not None:
                cv2.rectangle(img, (x1, y1), (x2, y2), colorB, border)

    # ตัวอักษร
    for i, text in enumerate(texts):
        if text not in ['None']:
            if type(text) is tuple:
                txt = text[0]
                colorT = text[1]
            else:
                txt = text
                colorT = (255, 255, 255)

            ox, oy = pos
            oy = int(oy + i * step)
            cv2.putText(img, txt, (ox, oy), font, scale, colorT, thickness)

    return img


def overlay(img_maino, img_overlay, pos: tuple = (0, 0)):
    '''
    Overlay function to blend an overlay image onto a main image at a specified position.

    :param img_main (numpy.ndarray): The main image onto which the overlay will be applied.
    :param img_overlay (numpy.ndarray): The overlay image to be blended onto the main image.
                                        IMREAD_UNCHANGED.
    :param pos (tuple): A tuple (x, y) representing the position where the overlay should be applied.

    :return: img_main (numpy.ndarray): The main image with the overlay applied in the specified position.
    '''
    img_main = img_maino.copy()
    if img_main.shape[2] == 4:
        img_main = cv2.cvtColor(img_main, cv2.COLOR_RGBA2RGB)

    x, y = pos
    h_overlay, w_overlay, _ = img_overlay.shape
    h_main, w_main, _ = img_main.shape

    x_start = max(0, x)
    x_end = min(x + w_overlay, w_main)
    y_start = max(0, y)
    y_end = min(y + h_overlay, h_main)

    img_main_roi = img_main[y_start:y_end, x_start:x_end]
    img_overlay_roi = img_overlay[(y_start - y):(y_end - y), (x_start - x):(x_end - x)]

    if img_overlay.shape[2] == 4:
        img_a = img_overlay_roi[:, :, 3] / 255.0
        img_rgb = img_overlay_roi[:, :, :3]
        img_overlay_roi = img_rgb * img_a[:, :, np.newaxis] + img_main_roi * (1 - img_a[:, :, np.newaxis])

    img_main_roi[:, :] = img_overlay_roi

    return img_main


def slope(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x1 - x2 == 0:
        m = math.inf
    else:
        m = (y1 - y2) / (x1 - x2)
    return m


def displacement(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))


def rotate(image, fee):
    if abs(fee) < 0.01:
        print(f"func rotate, fee={fee}, don't rotate, fee<0.01")
    else:
        print(f'func rotate, fee={fee}')
        image = ndimage.rotate(image, fee)
    return image


def rec2xy(rec):
    scope_xy1, scope_xy2 = rec
    scope_x1, scope_y1 = scope_xy1
    scope_x2, scope_y2 = scope_xy2
    return (scope_x1 + scope_x2) // 2, (scope_y1 + scope_y2) // 2,


def fine_mark(image, mark, rec, draw=None):
    scope_xy1, scope_xy2 = rec
    scope_x1, scope_y1 = scope_xy1
    scope_x2, scope_y2 = scope_xy2
    max_score = 0
    res = None

    result = cv2.matchTemplate(image[scope_y1:scope_y2, scope_x1:scope_x2], mark, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result > 0.7)
    for pt in zip(*loc):
        y, x = pt
        if max_score < result[y, x]:
            max_score = result[y, x]
            res = (scope_x1 + x, scope_y1 + y), (scope_x1 + x + mark.shape[1], scope_y1 + y + mark.shape[0])
    if draw is not None:
        cv2.rectangle(draw, *rec, (0, 255, 255), 2)
        if res:
            cv2.rectangle(draw, res[0], res[1], (0, 0, 255), 2)
    if res:
        return rec2xy(res)


def adj_image(image, frame):
    h, w, _ = image.shape
    rec1 = frame.marks['m1'].rec_around(h, w)
    rec2 = frame.marks['m2'].rec_around(h, w)
    mark1 = cv2.imread(f'data/{frame.name}/m1.png')
    mark2 = cv2.imread(f'data/{frame.name}/m1.png')
    m = slope(frame.marks['m1'].xypx(h, w), frame.marks['m2'].xypx(h, w))
    phi_default = round(math.degrees(math.atan(m)), 3)
    disp_default = round(displacement(frame.marks['m1'].xypx(h, w), frame.marks['m2'].xypx(h, w)), 3)

    point_m1 = fine_mark(image, mark1, rec1)
    point_m2 = fine_mark(image, mark2, rec2)

    if point_m1 is None or point_m2 is None:
        print('no mark point_be')
        return

    m = slope(point_m1, point_m2)
    phi = round(math.degrees(math.atan(m)), 3)
    disp = round(displacement(point_m1, point_m2), 3)
    image_ro = rotate(image, (phi - phi_default))
    # s = image_ro.copy()
    # s = cv2.resize(s, (0, 0), fx=0.4, fy=0.4)
    # cv2.imshow('image', s)
    # cv2.waitKey(0)
    draw = image_ro.copy()

    point_m1_af = fine_mark(image_ro, mark1, rec1, draw)
    point_m2_af = fine_mark(image_ro, mark2, rec2, draw)
    if point_m1_af is None or point_m2_af is None:
        print('no mark point_af <<<<<')
        return

    x = (frame.marks['m1'].xpx(h, w) - point_m1_af[0] + frame.marks['m2'].xpx(h, w) - point_m2_af[0]) // 2
    y = (frame.marks['m1'].ypx(h, w) - point_m1_af[1] + frame.marks['m2'].ypx(h, w) - point_m2_af[1]) // 2
    image = overlay(image, image_ro, (x, y))
    return image


if __name__ == '__main__':
    import cv2

    x = np.full((3000, 4000, 3), (0, 0, 100), dtype='uint8')
    cv2.imshow('nx', cv2.resize(x, (0, 0), fx=0.2, fy=0.2))
    cv2.waitKey(1)
    x = rotate(x, 10)
    cv2.imshow('nx', cv2.resize(x, (0, 0), fx=0.2, fy=0.2))
    cv2.waitKey(0)
