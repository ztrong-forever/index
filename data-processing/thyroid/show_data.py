import cv2
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt


data_path = "/data/guofeng/detection/Thyroid/data_4classes_new/images/val/"
label_path = "/data/guofeng/detection/Thyroid/data_4classes_new/labels/val/"
save_path = "/data/guofeng/detection/Thyroid/data_4classes_new/" + "show_val/"
if not os.path.exists(save_path):
    os.makedirs(save_path)


for image_name in tqdm.tqdm(os.listdir(data_path)):
    name1 = image_name.split(".")[0]
    imagePath = os.path.join(data_path, image_name)
    image = cv2.imread(imagePath)
    h0,w0,_ = image.shape

    for file in os.listdir(label_path):
        name2 = file.split(".")[0]
        if name1 == name2:
            labelPath = os.path.join(label_path, file)
            labels = np.loadtxt(labelPath)
            colors = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(0,255,255),(255,255,0)]

            if len(labels.shape) == 1:
                color = colors[0]
                label_class = labels[0]
                x = labels[1] * w0
                y = labels[2] * h0
                w = labels[3] * w0
                h = labels[4] * h0
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, str(int(label_class)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,0.8, color,2)
            else:
                for index, label in enumerate(labels):
                    color = colors[index]
                    label_class = label[0]
                    x = label[1] * w0
                    y = label[2] * h0
                    w = label[3] * w0
                    h = label[4] * h0
                    x1 = int(x - w/2)
                    y1 = int(y - h/2)
                    x2 = int(x + w/2)
                    y2 = int(y + h/2)
                    cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
                    cv2.putText(image, str(int(label_class)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,0.8, color,2)
            cv2.imwrite(save_path+image_name, image)


