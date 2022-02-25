import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import random
import tqdm

def read_imagesAndlabels(images_path, labels_path):

    images_list = []
    for image_path in os.listdir(images_path):
        images_list.append(os.path.join(images_path, image_path))

    labels_list = []
    for label_path in os.listdir(labels_path):
        labels_list.append(os.path.join(labels_path, label_path))

    return images_list, labels_list


def letterbox(img, width=1920, r=1.5, color=(0, 0, 0)):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]

    # Compute padding
    dw = width - shape[1]
    dh = width/r - shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # 判断dw和dh的大小
    if dh > 0 :
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    else :
        top, bottom = 0, 0

    if dw > 0:
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    else:
        left, right = 0, 0

    img_pad = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img_pad, (dw, dh)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    #    scale = [1, 1.2, 1.25, 4/3, 1.5, 16/9, 2]
    #     # scale =
    #     width = [1920, 1600, 1280, 1024, 800]

    width = 1920
    scale = 1

    images_path = "/home/data_4classes_mix/images/test/"
    labels_path = "/home/data_4classes_mix/labels/test/"
    save_images_path = "/home/resizeAndpad/pad/images/"
    save_labels_path = "/home/resizeAndpad/pad/labels/"

    images_list, labels_list = read_imagesAndlabels(images_path, labels_path)
    images_list = tqdm.tqdm(images_list)

    for index, image_path in enumerate(images_list):

        img = cv2.imread(image_path)
        h0, w0 = img.shape[:2]
        label_path = image_path.split("/")[-1].split(".")[0] + ".txt"
        labels = np.loadtxt(os.path.join(labels_path, label_path)).reshape(-1, 5)

        img_pad, (dw, dh) = letterbox(img, width=width, r=scale)

        if not os.path.exists(save_images_path):
            os.makedirs(save_images_path)

        if not os.path.exists(save_labels_path):
            os.makedirs(save_labels_path)

        cv2.imwrite(save_images_path + image_path.split("/")[-1], img_pad)

        dw, wh = (dw, dh)

        for i in range(len(labels)):

            # 还原原始图像的x,y,w,h
            x = labels[i, 1] * w0
            y = labels[i, 2] * h0
            w = labels[i, 3] * w0
            h = labels[i, 4] * h0

            # 计算原始图像左上角坐标和有下角坐标
            ay = int(y - h / 2)
            ax = int(x - w / 2)
            by = int(y + h / 2)
            bx = int(x + w / 2)

            # 计算改变后的labels

            x_pad = int(x + dw)
            y_pad = int(y + dh)
            w_pad = int(w)
            h_pad = int(h)

            ay_pad = int(y_pad - h_pad / 2)
            ax_pad = int(x_pad - w_pad / 2)
            by_pad = int(y_pad + h_pad / 2)
            bx_pad = int(x_pad + w_pad / 2)

            with open(save_labels_path + label_path,"a") as f:
                f.write(" {0} {1} {2} {3} {4} \n".format(labels[i,0], x_pad, y_pad, w_pad, h_pad))

            # cv2.rectangle(img, (ax, ay), (bx, by), (255, 0, 0), 2)
            # cv2.rectangle(img_pad,(ax_pad,ay_pad),(bx_pad,by_pad),(255,0,0),2)



