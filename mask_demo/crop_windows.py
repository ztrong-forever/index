import os
import cv2
import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt


# def show_image(image, cropped_image, top, bottom, top_x, top_y, bottom_x, bottom_y):
#
#     cv2.circle(cropped_image, (top_x + 10, top_y - 50), 1, (0, 0, 255), 5)
#     cv2.circle(cropped_image, (bottom_x + 10, bottom_y - 50), 1, (0, 0, 255), 5)
#     # cv2.circle(image, (int(top[0]), int(top[1])), 1, (0, 0, 255), 5)
#     # cv2.circle(image, (int(bottom[0]) , int(bottom[1]) ), 1, (0, 0, 255), 5)
#     # plt.imshow(image)
#     # plt.show()
#     plt.imshow(cropped_image)
#     plt.show()



def read_point_json(key_point_path, imagename):

    imagename = imagename.split(".")[0] + ".json"
    json_file = os.path.join(key_point_path, imagename)

    with open(json_file, 'r') as j:
        json_data = json.load(j)
        shapes_info = json_data['shapes']
        if shapes_info[0]["label"] == "top":
            top = shapes_info[0]["points"][0]
        else:
            bottom = shapes_info[0]["points"][0]

        if shapes_info[1]["label"] == "bottom":
            bottom = shapes_info[1]["points"][0]
        else:
            top = shapes_info[1]["points"][0]

    return top, bottom



def transformer_point(image, cropped_image, key_point, crop):

    h0, w0, _ = image.shape
    h, w, _ = cropped_image.shape
    x1, y1, x2, y2 = crop
    top, bottom = key_point
    top_x = int(top[0] - x1)
    top_y = int(top[1] - y1)
    bottom_x = int(bottom[0] - x1)
    bottom_y = int(bottom[1] - y1)

    # show_image(image, cropped_image, top,bottom, top_x, top_y, bottom_x, bottom_y)

    # GE
    # return top_x + 10, top_y - 50, bottom_x + 10, bottom_y - 50

    # PHLIPHS
    return top_x, top_y, bottom_x, bottom_y




def save_point_txt(imagename, top_x, top_y, bottom_x, bottom_y, save_point_path):

    with open(save_point_path + imagename.split(".")[0] + ".txt", "a+") as f:
        top = " ".join(("top", str(top_x), str(top_y)))
        bottom = " ".join(("bottom", str(bottom_x), str(bottom_y)))
        txt_info = ' '.join((imagename, top, bottom))
        f.writelines(txt_info + '\n')




path = "C:\\Users\\Administrator\\Desktop\\result\\GE\\GE"
key_point_path = "C:\\Users\\Administrator\\Desktop\\result\\GE\\keypoint_jsons\\"
save_path = "C:\\Users\\Administrator\\Desktop\\result\\GE\\GE_save\\"
error_path = "C:\\Users\\Administrator\\Desktop\\result\\GE\\GE_error\\"
save_point_path = "C:\\Users\\Administrator\\Desktop\\result\\GE\\GE_point_txt\\"

# path = "C:\\Users\\Administrator\\Desktop\\result\\GE\\GE_mask"
# key_point_path = "C:\\Users\\Administrator\\Desktop\\result\\GE\\keypoint_jsons\\"
# save_path = "C:\\Users\\Administrator\\Desktop\\result\\GE\\GE_mask_save\\"
# error_path = "C:\\Users\\Administrator\\Desktop\\result\\GE\\GE_mask_error\\"
# # save_point_path = "C:\\Users\\Administrator\\Desktop\\result\\GE\\GE_point_txt\\"

# path = "C:\\Users\\Administrator\\Desktop\\result\\PHLIPHS\\PHLIPHS_mask"
# key_point_path = "C:\\Users\\Administrator\\Desktop\\result\\PHLIPHS\\keypoint_jsons\\"
# save_path = "C:\\Users\\Administrator\\Desktop\\result\\PHLIPHS\\PHLIPHS_mask_save\\"
# error_path = "C:\\Users\\Administrator\\Desktop\\result\\PHLIPHS\\PHLIPHS_mask_error\\"
# # save_point_path = "C:\\Users\\Administrator\\Desktop\\result\\PHLIPHS\\PHLIPHS_txt\\"

# path = "C:\\Users\\Administrator\\Desktop\\result\\PHLIPHS\\PHLIPHS"
# key_point_path = "C:\\Users\\Administrator\\Desktop\\result\\PHLIPHS\\keypoint_jsons\\"
# save_path = "C:\\Users\\Administrator\\Desktop\\result\\PHLIPHS\\PHLIPHS_save\\"
# error_path = "C:\\Users\\Administrator\\Desktop\\result\\PHLIPHS\\PHLIPHS_error\\"
# save_point_path = "C:\\Users\\Administrator\\Desktop\\result\\PHLIPHS\\PHLIPHS_point_txt\\"

for file in tqdm.tqdm(os.listdir(path)):

    filename = file.split(".")[0]
    filename_img = file.split(".")[-1]

    if filename_img != "txt":
        img_path = os.path.join(path, file)
        txt_path = os.path.join(path, filename + ".txt")
        image = cv2.imread(img_path)
        h0, w0, _ = image.shape

        try:
            windows = np.loadtxt(txt_path)
            if len(windows) == 5:
                x = windows[1]
                y = windows[2]
                w = windows[3]
                h = windows[4]
                if x != 0 and y != 0 and w != 0 and h != 0:
                    x = x * w0
                    y = y * h0
                    w = w * w0
                    h = h * h0
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)
                    crop = [x1, y1, x2, y2]
                    # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # plt.imshow(image)
                    # plt.show()
                    cropped = image[y1+50:y2, x1-10:x2+10]
                    # cropped = image[y1:y2, x1:x2]
                    top, bottom = read_point_json(key_point_path, file)
                    key_point = [top, bottom]
                    top_x, top_y, bottom_x, bottom_y = transformer_point(image, cropped, key_point, crop)
                    save_point_txt(file,top_x, top_y, bottom_x, bottom_y,save_point_path)
                    cv2.imwrite(save_path + file, cropped)


            else:
                for box in windows:
                    x = box[1]
                    y = box[2]
                    w = box[3]
                    h = box[4]
                    if x != 0 and y != 0 and w != 0 and h != 0:
                        x = x * w0
                        y = y * h0
                        w = w * w0
                        h = h * h0
                        x1 = int(x - w / 2)
                        y1 = int(y - h / 2)
                        x2 = int(x + w / 2)
                        y2 = int(y + h / 2)
                        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # plt.imshow(image)
                        # plt.show()
                        crop = [x1, y1, x2, y2]
                        # cropped = image[y1:y2,x1:x2]

                        cropped = image[y1 + 50:y2, x1 - 10:x2 + 10]
                        top, bottom = read_point_json(key_point_path, file)
                        key_point = [top, bottom]
                        top_x, top_y, bottom_x, bottom_y = transformer_point(image, cropped, key_point, crop)
                        save_point_txt(file, top_x, top_y, bottom_x, bottom_y, save_point_path)
                        cv2.imwrite(save_path + file, cropped)


        except:
            cv2.imwrite(error_path + file, image)
            pass




