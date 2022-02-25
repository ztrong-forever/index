# import os
# import json
# import cv2 as cv
# import numpy as np
#
#
# path = "C:\\Users\\Administrator\\Desktop\\img"
#
# for filename in os.listdir(path):
#
#     if filename.split(".")[-1] == "json":
#         with open("C:\\Users\\Administrator\\Desktop\\img\\"+filename, 'r') as load_f:
#             load_dict = json.load(load_f)
#         points = load_dict["shapes"][0]
#         points = points["points"]
#         points = points
#         image = cv.imread("C:\\Users\\Administrator\\Desktop\\img\\" + filename.split(".")[0] + ".jpg")
#
#         # imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#         # ret, thresh = cv.threshold(imgray, 127, 255, 0)
#         # image, contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
#         temp = np.zeros(image.shape, np.uint8)
#
#         # 原本thickness = -1表示内部填充
#         mask = cv.fillPoly(image, points, color=(255, 255, 255))
#         cv.imwrite("C:\\Users\\Administrator\\Desktop\\" + "1.jpg",mask)
#         break


# import json
# import os
# import os.path as osp
# import warnings
# from shutil import copyfile
# import PIL.Image
# import yaml
# from labelme import utils
# import time
#
#
# def main():
#     json_file = 'C:\\Users\\Administrator\\Desktop\\img_json'
#
#     list = os.listdir(json_file)
#     if not os.path.exists(json_file + '/' + 'pic'):
#         os.makedirs(json_file + '/' + 'pic')
#     if not os.path.exists(json_file + '/' + 'cv_mask'):
#         os.makedirs(json_file + '/' + 'cv_mask')
#     if not os.path.exists(json_file + '/' + 'labelme_json'):
#         os.makedirs(json_file + '/' + 'labelme_json')
#     if not os.path.exists(json_file + '/' + 'json'):
#         os.makedirs(json_file + '/' + 'json')
#
#     for i in range(0, len(list)):
#
#         path = os.path.join(json_file, list[i])
#         if os.path.isfile(path):
#
#             copyfile(path, json_file + '/json/' + list[i])
#             data = json.load(open(path))
#             img = utils.img_b64_to_arr(data['imageData'])
#             lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])
#
#             captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
#             lbl_viz = utils.draw_label(lbl, img, captions)
#             out_dir = osp.basename(list[i]).replace('.', '_')
#             out_dir = osp.join(osp.dirname(list[i]), out_dir)
#
#             filename = out_dir[:-5]
#
#             out_dir = json_file + "/" + 'labelme_json' + "/" + out_dir
#             out_dir1 = json_file + "/" + 'pic'
#             out_dir2 = json_file + "/" + 'cv_mask'
#
#             if not osp.exists(out_dir):
#                 os.mkdir(out_dir)
#
#             print(img)
#             PIL.Image.fromarray(img).save(osp.join(out_dir, 'img' + '.jpg'))
#             PIL.Image.fromarray(img).save(osp.join(out_dir1, str(filename) + '.jpg'))
#
#             utils.lblsave(osp.join(out_dir, 'label.jpg'), lbl)
#             utils.lblsave(osp.join(out_dir2, str(filename) + '.png'), lbl)
#
#         PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.jpg'))
#
#         with open(osp.join(out_dir, 'label_names' + '.txt'), 'w') as f:
#             for lbl_name in lbl_names:
#                 f.write(lbl_name + '\n')
#
#         warnings.warn('info.yaml is being replaced by label_names.txt')
#         info = dict(label_names=lbl_names)
#         with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
#             yaml.dump(info, f, default_flow_style=False)
#
#         fov = open(osp.join(out_dir, 'info' + '.yaml'), 'w')
#         for key in info:
#             fov.writelines(key)
#             fov.write(':\n')
#         for k, v in lbl_names.items():
#             fov.write('-')
#             fov.write(' ')
#             fov.write(k)
#             fov.write('\n')
#
#         fov.close()
#         print('Saved to: %s' % out_dir)
#
#
# if __name__ == '__main__':
#     start = time.time()
#     main()
#     spend = time.time() - start
#     print(spend)


import os
import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = "C:\\Users\\Administrator\\Desktop\\img_json\\cv_mask\\"
for filename in os.listdir(path):
    img_path = path + filename
    img = cv.imread(img_path,0)
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.fillPoly(img, contours, (255, 255, 255))
    cv.imwrite("C:\\Users\\Administrator\\Desktop\\img_mask\\" + filename.split(".")[0]+".jpg",img)
