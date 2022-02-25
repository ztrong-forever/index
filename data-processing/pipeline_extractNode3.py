'''
Description: Shanghai Xingmai Information Technology Co., Ltd.
Autor: yeqy
Date: 2021-09-24 05:36:17
LastEditors: yeqy
LastEditTime: 2021-09-30 08:21:35
FilePath: /ThyroidPipeline-1.0/pipeline_extractNode.py
'''
import os
import numpy as np
import cv2
from glob import glob
import random
import shutil


class pipline_extractNode:

    def __init__(self, video_paths="", txt_dirs="", output_dir=""):
        self.video_paths = video_paths
        self.txt_dirs = txt_dirs
        self.output_dir = output_dir

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        '''
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
        label = '%s %.2f' % (names[int(cls)], conf)
        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
        '''
        # Plots one bounding box on image img
        tl = line_thickness or round(
            0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def xyxy2xywh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = np.zeros_like(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    def PointsNearby(self, point, points, threshold):
        """
        args:
            point:    1x4
            points:   nx4
            t_rate:   1(float)
        """
        point = np.array(point)
        dist = np.linalg.norm(point[-4:-2] - points[-1, -4:-2])
        # refine threshold by size of bbox(width)
        # points_LastSize = np.mean(points[-1, 5:7])
        point_size = np.mean(point[-2:])
        return np.min(dist) < threshold * point_size

    def SizeSimilar(self, point, points, threshold):
        point = np.array(point)
        point_size = np.mean(point[-4:-2])
        points_LastSize = np.mean(points[-1, -4:-2])
        rate = np.abs((point_size - points_LastSize) / points_LastSize)

        return rate < threshold

    def FrameNearby(self, point, points):
        point = np.array(point)
        point_diff = point[0] - points[-1, 0]

        points_LastSize = np.mean(points[-1, -4:-2])
        if points_LastSize < 50:
            threshold = 15
        elif 50 <= points_LastSize < 80:
            threshold = 20
        elif 80 <= points_LastSize < 160:
            threshold = 25
        else:
            threshold = 30

        return point_diff < threshold

    def collect_yolo_output(self, video_path, txt_dir, cls_name=0):
        """
        return:
            [frame_id, cls, conf, x, y, w, h]
        """
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        video_name = os.path.basename(video_path).split('.')[0]
        txt_paths = glob(txt_dir + '/' + video_name + '*.txt')
        # txt_paths = txt_dir
        output_infos = []
        for path in txt_paths:
            frame_id = path.split('_')[1].split('.')[0]
            with open(path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                bbox_info_list = line.strip().split(' ')
                # 读取指定类别，没有指定则全部读取
                if int(bbox_info_list[0]) == cls_name or cls_name == None:
                    output_infos.append([frame_id] + bbox_info_list)
        output_infos = np.array(output_infos).astype(float)

        # 将归一化的Bbox转化为实际图像尺寸对应的Bbox
        xywhs = (output_infos[:, -4:] * np.array([w, h, w, h])).astype(int)
        output_infos = np.concatenate([output_infos[:, :-4], xywhs], axis=1)
        # 根据第一列即frame_id排序，升序
        output_infos = output_infos[np.argsort(output_infos[:, 0])]
        return output_infos

    def NodeClassification(self, detected_infos):
        """根据Bbox坐标、尺寸以及帧间隔划分结节"""
        # [frame_id, cls, conf, x, y, w, h]
        cls_infos = []
        for bbox in detected_infos:
            if len(cls_infos) == 0:
                cls_infos.append([bbox])
            else:
                SameFlag = False
                for i in range(len(cls_infos) - 1, -1, -1):
                    bbox_group = np.array(cls_infos[i])
                    is_LocationNearby = self.PointsNearby(bbox, bbox_group, threshold=0.5)
                    is_SizeNearby = self.SizeSimilar(bbox, bbox_group, threshold=0.8)
                    is_FrameNearby = self.FrameNearby(bbox, bbox_group)

                    if is_LocationNearby and is_SizeNearby and is_FrameNearby:
                        cls_infos[i].append(bbox)
                        SameFlag = True
                        break

                if not SameFlag:
                    cls_infos.append([bbox])

        return cls_infos

    def rmFewFrame(slef, cls_infos, f_rate):
        frame_thres = [int(f_rate * np.mean(np.array(i)[:, -2:])) for i in cls_infos]
        cls_infos = [cls for i, cls in enumerate(cls_infos) if len(cls) > frame_thres[i]]
        return cls_infos

    def selectMainObject(self, cls_infos, method='area'):
        objectNum = len(cls_infos)
        returnObj = []
        for i in range(objectNum):
            obj = np.array(cls_infos[i])
            if method == 'area':
                obj_area = obj[:, -4] * obj[:, -3]
                maxAreaIdx = np.argmax(obj_area)
                returnObj.append(obj[maxAreaIdx])

        return returnObj

    def result(self):

        sum_node = 0

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        for video_path in os.listdir(self.video_paths):
            video_root = os.path.join(self.video_paths, video_path)
            # print(video_root)

            for txt_dir in os.listdir(self.txt_dirs):
                if "txt" in txt_dir:
                    if txt_dir.split("_")[0] == video_path.split(".")[0]:
                        txt_dir_root = os.path.join(self.txt_dirs, txt_dir)
                        # print(txt_dir_root)
                        if not os.path.exists(os.path.join(self.txt_dirs, txt_dir.split("_")[0])):
                            os.mkdir(os.path.join(self.txt_dirs, txt_dir.split("_")[0]))
                        shutil.copy(txt_dir_root, os.path.join(self.txt_dirs, txt_dir.split("_")[0]))

            for root, dirs, files in os.walk(self.txt_dirs):
                if video_path.split(".")[0] == root.split("/")[-1]:

                    # try:
                    output_infos = self.collect_yolo_output(video_root, root + "/", cls_name=0)
                    cls_infos = self.NodeClassification(output_infos)
                    # cls_infos = rmFewFrame(cls_infos, f_rate=0.01)
                    returnObj = self.selectMainObject(cls_infos)
                    cap = cv2.VideoCapture(video_root)
                    sum_node += len(returnObj)

                    for i, obj in enumerate(returnObj):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, obj[0])  # 设置要获取的帧号
                        ret, frame = cap.read()

                        if ret:
                            xyxy = self.xywh2xyxy(obj[-4:].reshape((-1, 4))).flatten()
                            self.plot_one_box(xyxy, frame, label=str(i))
                            video_name = os.path.basename(video_root).split('.')[0]
                            SaveImgPath = os.path.join(self.output_dir,
                                                       video_name + str(int(obj[0])) + '_' + str(i) + '.jpg')
                            cv2.imwrite(SaveImgPath, frame)

                    cap.release()

                    # except Exception as e:

                        # print("errors,Please check:", video_root, "\t", root, "!")

        return sum_node


if __name__ == "__main__":
    video_paths = "/data/guofeng/detection/Thyroid/testData/testVideos/"
    txt_dirs = "/home/projects/yolov5-3.1/inference/thyroid/exp34_test/"
    output_dir = "/home/projects/yolov5-3.1/inference/thyroid/exp34_test-n"
    extractNode = pipline_extractNode(video_paths, txt_dirs, output_dir)
    print(extractNode.sum_node)
