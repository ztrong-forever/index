# coding: utf-8
import os
import random
import shutil
import cv2

random.seed(20)

def split(id_list, rate):
    random.shuffle(id_list)
    trainnum = int(rate[0] * n)
    testnum = int(rate[1] * n)
    valnum = int(rate[2] * n)
    train_id = id_list[:trainnum]
    test_id = id_list[trainnum:trainnum + testnum]
    val_id = id_list[trainnum + testnum:]
    return train_id, test_id, val_id

def shutil_data(img_dir, lb_dir, image, label):
    shutil.copy(image, img_dir)
    shutil.copy(label, lb_dir)


if __name__ == '__main__':
    # data_dir = "C:\\Users\\Administrator\\Desktop\\window_heart\\PHLIPHS\\images\\"
    # labels_dir = "C:\\Users\\Administrator\\Desktop\\window_heart\\PHLIPHS\\labels\\"

    data_dir = "C:\\Users\\Administrator\\Desktop\\window_heart\\GE\\images\\"
    labels_dir = "C:\\Users\\Administrator\\Desktop\\window_heart\\GE\\labels\\"

    save_image_dir = "C:\\Users\\Administrator\\Desktop\\window_heart\\"
    save_label_dir = "C:\\Users\\Administrator\\Desktop\\window_heart\\"

    global filelist
    filelist = os.listdir(data_dir)
    id_list = []
    for filename in filelist:
        id_list.append(filename)
    id_list = list(set(id_list))
    n = len(id_list)

    rate = [0.8, 0.1, 0.1]
    train_ids, test_ids, val_ids = split(id_list, rate)

    ## train
    for train_id in train_ids:
        ## image
        image_path = os.path.join(data_dir, train_id)
        train_dir = os.path.join(save_image_dir, "images\\train")

        ## label
        label = train_id.split(".")[0] + ".txt"
        label_path = os.path.join(labels_dir, label)
        label_dir = os.path.join(save_label_dir, "labels\\train")

        shutil_data(train_dir, label_dir, image_path, label_path)

    ## test
    for test_id in test_ids:
        ## image
        image_path = os.path.join(data_dir, test_id)
        train_dir = os.path.join(save_image_dir, "images\\test")

        ## label
        label = test_id.split(".")[0] + ".txt"
        label_path = os.path.join(labels_dir, label)
        label_dir = os.path.join(save_label_dir, "labels\\test")

        shutil_data(train_dir, label_dir, image_path, label_path)

    ## val
    for val_id in val_ids:
        ## image
        image_path = os.path.join(data_dir, val_id)
        train_dir = os.path.join(save_image_dir, "images\\val")

        ## label
        label = val_id.split(".")[0] + ".txt"
        label_path = os.path.join(labels_dir, label)
        label_dir = os.path.join(save_label_dir, "labels\\val")

        shutil_data(train_dir, label_dir, image_path, label_path)








