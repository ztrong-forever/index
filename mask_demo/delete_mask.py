import cv2
import os
import numpy as np


file_path = "C:\\Users\\Administrator\\Desktop\\delete_output\\"
out_path = "C:\\Users\\Administrator\\Desktop\\delete_output2\\"

for filename in os.listdir(file_path):
    file = file_path + filename
    img = cv2.imread(file,0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    area = []

    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))

    max_idx = np.argmax(area)
    img = np.zeros(img.shape, dtype=np.uint8)
    max_area = cv2.contourArea(contours[max_idx])

    for k in range(len(contours)):
        # cv2.drawContours(img, [contours[k]], 0, (255, 255, 255), 2)
        if k == max_idx:
            # cv2.fillPoly(img, [contours[k]], 255)
            cv2.drawContours(img,contours[k],0,(255,255,255),2)


    # img[img < 255] = 0

    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if img[i, j] == 255:
    #             continue
    #         else:
    #             img[i, j] = 0

    # if len(contours) > 100:
    #     # for i in range(1, len(contours)):
    #     #     # if contours[0].shape[0] / contours[i].shape[0] < 6 or contours[0].shape[0] / contours[i].shape[0] == 6:
    #     #     cv2.fillPoly(mask_sel, [contours[i]], 255)
    #     #     mask_sel = cv2.erode(mask_sel, kernel, iterations=1)
    #     # cv2.imwrite(os.path.join("/home/data/findmask2/" + dir, image_name), mask_sel)
    #     continue
    #
    # else:
    cv2.imwrite(out_path + filename, img)
