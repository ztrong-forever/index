
# coding: utf-8
# auhtor: hxy

import os
import cv2
from tqdm import tqdm

img_folder = "C:\\Users\\Administrator\\Desktop\\result\\PHLIPHS\\PHLIPHS_mask_save\\"
point_txt = "C:\\Users\\Administrator\\Desktop\\result\\PHLIPHS\\point.txt"


with open(point_txt, 'r') as f:
    infos = f.readlines()
    for info in tqdm(infos):
        info = info.strip('\n')
        info_data = info.split(' ')
        img_name = info_data[0]

        top =[int(float(info_data[2])), int(float(info_data[3]))]
        bottom = [int(float(info_data[5])), int(float(info_data[6]))]

        img_full_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_full_path)

        img = cv2.circle(img, (top[0], top[1]), 3, (0,0,255), -1)
        img = cv2.circle(img, (bottom[0], bottom[1]), 3, (0,0,255), -1)

        cv2.imwrite(os.path.join("C:\\Users\\Administrator\\Desktop\\output2\\", img_name), img)


