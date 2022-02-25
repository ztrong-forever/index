import os
import cv2


image_dir = "C:\\Users\\Administrator\\Desktop\\window_heart\\images\\val"
label_dir = "C:\\Users\\Administrator\\Desktop\\window_heart\\labels\\val"

for image in os.listdir(image_dir):
    img = os.path.join(image_dir, image)
    labels = image.split(".")[0] + ".txt"
    labels = os.path.join(label_dir, labels)

    img = cv2.imread(img)
    h0, w0, _ = img.shape
    with open(labels,"r") as f:
        label = f.readlines()[0]
        label = label.strip("\n").split(" ")
        x = float(label[1]) * w0
        y = float(label[2]) * h0
        w = float(label[3]) * w0
        h = float(label[4]) * h0

        x1 = int(x - 0.5 * w)
        y1 = int(y - 0.5 * h)
        x2 = int(x + 0.5 * w)
        y2 = int(y + 0.5 * h)

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.imshow("img",img)
        cv2.waitKey()