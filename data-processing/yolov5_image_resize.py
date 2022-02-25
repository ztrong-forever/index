import cv2
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random


def read_imagesAndlabels(images_path, labels_path):

    images_list = []
    for image_path in os.listdir(images_path):
        images_list.append(os.path.join(images_path, image_path))

    labels_list = []
    for label_path in os.listdir(labels_path):
        labels_list.append(os.path.join(labels_path, label_path))

    return images_list, labels_list



def resize_image():
    pass

if __name__ == "__main__":

    # 1920 1080

    parser = argparse.ArgumentParser()

    images_path = "/home/data_4classes_mix/images/test/"
    labels_path = "/home/data_4classes_mix/labels/test/"

    images_list, labels_list = read_imagesAndlabels(images_path, labels_path)

    for index, image_path in enumerate(images_list):
        chance = random.uniform(0,1)
        img = cv2.imread(image_path)
        h0, w0 = img.shape[:2]
        label_path = image_path.split("/")[-1].split(".")[0] + ".txt"
        labels = np.loadtxt(os.path.join(labels_path,label_path)).reshape(1,-1)

        # 还原原始图像的x,y,w,h
        x = labels[0,1] * w0
        y = labels[0,2] * h0
        w = labels[0,3] * w0
        h = labels[0,4] * h0

        # 计算原始图像左上角坐标和有下角坐标
        ay = int(y - h/2)
        ax = int(x - w/2)
        by = int(y + h/2)
        bx = int(x + w/2)

        # resize插值
        if chance > 0 and chance < 0.25:
            resize_img = cv2.resize(img, dsize=(1920,1080), interpolation=cv2.INTER_NEAREST)
        elif chance > 0.25 and chance < 0.5:
            resize_img = cv2.resize(img, dsize=(1920,1080), interpolation=cv2.INTER_LINEAR)
        elif chance > 0.5 and chance < 0.75:
            resize_img = cv2.resize(img, dsize=(1920,1080), interpolation=cv2.INTER_CUBIC)
        else:
            resize_img = cv2.resize(img, dsize=(1920,1080), interpolation=cv2.INTER_AREA)

        # resize box
        re_x = x * 1920/w0
        re_y = y * 1080/h0
        re_w = w * 1920/w0
        re_h = h * 1080/h0

        # 计算resize图像左上角坐标和有下角坐标
        re_ay = int(re_y - re_h / 2)
        re_ax = int(re_x - re_w / 2)
        re_by = int(re_y + re_h / 2)
        re_bx = int(re_x + re_w / 2)

        print("test")