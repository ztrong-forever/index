import os
import cv2

file_path = "C:\\Users\\Administrator\\Desktop\\window_heart\\PHLIPHS\\phliphs_window_loc.txt"
image_path = "C:\\Users\\Administrator\\Desktop\\window_heart\\PHLIPHS\\images\\"
label_path = "C:\\Users\\Administrator\\Desktop\\window_heart\\PHLIPHS\\labels\\"

# file_path = "C:\\Users\\Administrator\\Desktop\\window_heart\\GE\\ge_window_loc.txt"
# image_path = "C:\\Users\\Administrator\\Desktop\\window_heart\\GE\\images\\"
# label_path = "C:\\Users\\Administrator\\Desktop\\window_heart\\GE\\labels\\"

with open(file_path,"r") as f:
    datas = f.readlines()

for data in datas:
    data = data.replace(",", " ").strip("\n").split(" ")

    img = os.path.join(image_path, data[0])
    image = cv2.imread(img)
    h0, w0, _ = image.shape

    filename = data[0].split(".")[0] + ".txt"
    x1 = float(data[1])
    y1 = float(data[2])
    x2 = float(data[3])
    y2 = float(data[4])

    x = (x1 + 0.5 * (x2 - x1)) / w0
    y = (y1 + 0.5 * (y2 - y1)) / h0
    w = (x2 - x1) / w0
    h = (y2 - y1) / h0

    with open(label_path + filename, "w") as f:

        f.write("{} {} {} {} {}\n".format(0, x, y, w, h))

