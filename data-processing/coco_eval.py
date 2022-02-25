import os
import json
import glob
import cv2
import shutil
import numpy as np
from coco import COCO
from cocoeval import COCOeval
from pipeline_extractNode3 import pipline_extractNode as extractNode


class Metrics(object):
    def __init__(self, cocoGt_file, cocoDt_file, conf_thres_list):
        self.cocoGt_file = cocoGt_file
        self.cocoDt_file = cocoDt_file
        self.conf_thres_list = conf_thres_list

    @staticmethod
    def convert(size, box):
        """将ROI的坐标转换为yolo需要的坐标
        :param size: size是图片的w和h
        :param box: box里保存的是ROI的坐标（x，y的最大值和最小值）
        :return: 返回值为ROI中心点相对于图片大小的比例坐标，和ROI的w、h相对于图片大小的比例
        """
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = box[0] + box[2] / 2.0
        y = box[1] + box[3] / 2.0
        w = box[2]
        h = box[3]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    @staticmethod
    def calc_iou(bbox1, bbox2):
        if not isinstance(bbox1, np.ndarray):
            bbox1 = np.array(bbox1)
        if not isinstance(bbox2, np.ndarray):
            bbox2 = np.array(bbox2)
        xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
        xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)

        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

        ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
        xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
        ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
        xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))

        h = np.maximum(ymax - ymin, 0)
        w = np.maximum(xmax - xmin, 0)
        intersect = h * w

        union = area1 + np.squeeze(area2, axis=-1) - intersect
        return intersect / union

    def fiter_predict_results(self, conf_thres=0.001):
        filted_results = []
        with open(self.cocoDt_file, "r") as jf:
            det_results = json.load(jf)

            for det in det_results:
                if det["score"] >= conf_thres:
                    filted_results.append(det)
        return filted_results

    def compute_coco_map(self, imgid=None, catIds=[0], report=False):
        cocoGt = COCO(self.cocoGt_file)  # initialize COCO ground truth api
        imgIds = cocoGt.getImgIds()
        cocoDt = cocoGt.loadRes(self.cocoDt_file)  # initialize COCO pred api
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        if imgid is None:
            cocoEval.params.imgIds = imgIds  # image IDs to evaluate
        else:
            cocoEval.params.imgIds = imgid  # image IDs to evaluate
        cocoEval.params.catIds = catIds
        cocoEval.params.report = report
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        ap50 = cocoEval.stats[0]
        tp = int(cocoEval.stats[8])
        fp = int(cocoEval.stats[9])
        fn = int(cocoEval.stats[10])

        p = tp / (tp + fp + 1e-16)
        r = tp / (tp + fn + 1e-16)

        f1 = 2 * p * r / (p + r + 1e-16)
        return ap50, tp, fp, fn, f1, p

    def compute_nodule_f1(self):
        best_f1 = 0
        best_conf = 0
        best_fp = np.inf
        out_str = ""
        for conf_thres in self.conf_thres_list:
            if conf_thres == 0.001:
                report = True
            else:
                report = False
            cocoDt_res = self.fiter_predict_results(conf_thres=conf_thres)
            self.cocoDt_file = cocoDt_res
            ap50, tp, fp, fn, f1, p = self.compute_coco_map(report=report)
            # print("conf thres:{:.3f}, ap50:{:.3f}, tp:{}, fp:{}, fn:{}, f1:{:.4f}, p:{:.4f}".format(
            #     conf_thres, ap50, tp, fp, fn, f1, p))

            if f1 >= best_f1:
                best_f1 = f1
                # best_conf = conf_thres
                out_str = "F1:{:.3f}, TP:{}, FP:{}, FN:{}".format(f1, tp, fp, fn)
            if fp <= best_fp:
                best_fp = fp

        # print("best conf thres:{:.3f}, best f1 score:{:.4f}, best fp:{}".format(best_conf, best_f1, best_fp))
        print("Image level metric: {}".format(out_str))

    def parse_json(self, conf_thres=0.001):
        gt_info, dt_info = {}, {}

        # 加载预测结果
        with open(self.cocoDt_file, "r") as jf:
            det_results = json.load(jf)
            for det in det_results:
                if det["image_id"] not in dt_info.keys():
                    dt_info[det["image_id"]] = {"boxes": [], "scores": []}
                if det["score"] >= conf_thres:
                    det["bbox"][2] = det["bbox"][0] + det["bbox"][2]
                    det["bbox"][3] = det["bbox"][1] + det["bbox"][3]
                    dt_info[det["image_id"]]["boxes"].append(det["bbox"])
                    dt_info[det["image_id"]]["scores"].append(det["score"])

        # 加载 CoCo 标签数据
        coco = COCO(self.cocoGt_file)
        imgIds = coco.getImgIds()
        for imgId in imgIds:
            img = coco.loadImgs(imgId)[0]
            # image_name = img['file_name']

            annIds = coco.getAnnIds(imgIds=img['id'], catIds=[], iscrowd=None)
            anns = coco.loadAnns(annIds)

            if len(anns):
                if img['id'] not in gt_info.keys():
                    gt_info[img['id']] = {"boxes": [], "labels": []}
                for ann in anns:
                    ann["bbox"][2] = ann["bbox"][0] + ann["bbox"][2]
                    ann["bbox"][3] = ann["bbox"][1] + ann["bbox"][3]
                    gt_info[img['id']]["boxes"].append(ann["bbox"])
                    gt_info[img['id']]["labels"].append(ann["category_id"])

        return gt_info, dt_info

    def compute_metrics(self, gt_info, dt_info, iou_thres=0.5, num_thres=1, num_nodule=25):
        det_res = {}
        for i in range(num_nodule):
            det_res[i] = 0

        for imgid, img_info in gt_info.items():
            if imgid in dt_info.keys():
                if not len(dt_info[imgid]["boxes"]): continue

                ious = self.calc_iou(img_info["boxes"], dt_info[imgid]["boxes"])
                idx = np.where(ious >= iou_thres)
                left_idx = idx[0]
                right_idx = idx[1]

                if len(right_idx) and len(left_idx):
                    for li in left_idx:
                        det_res[img_info["labels"][li]] += 1

                    for ri in reversed(sorted(right_idx)):
                        del dt_info[imgid]["boxes"][ri]
                        del dt_info[imgid]["scores"][ri]

        tp = 0
        absence_nodule = []
        for noduleid, num in det_res.items():
            if num >= num_thres:
                tp += 1
            else:
                absence_nodule.append(noduleid)
        return tp, tp / num_nodule, absence_nodule, dt_info

    def compute_nodule_acc(self, iou_thres=0.5, num_thres=1, num_nodule=25):
        data_info = {}
        for conf_thres in self.conf_thres_list:
            gt_info, dt_info = self.parse_json(conf_thres=conf_thres)
            tp, acc, abs_nodule, dt_info = self.compute_metrics(gt_info, dt_info, iou_thres, num_thres, num_nodule)
            # print("conf thres:{:.3f}, acc:{:.3f}".format(conf_thres, acc))

            data_info[acc] = {"conf": 0, "tp": 0, "node": [], "dt_info": {}}
            data_info[acc]["conf"] = conf_thres
            data_info[acc]["tp"] = tp
            data_info[acc]["node"] = abs_nodule
            data_info[acc]["dt_info"] = dt_info

        # print("max conf thres:{:.3f}, max acc:{:.3f}, absence nodule:{}".format(max_conf, max_acc, absence))

        return data_info

    def save_det_results(self, dt_info, save_path, acc=0.1, save_img=False,
                         save_txt=False, coco_path="", task="test"):
        img_path = os.path.join(coco_path, "images", task)
        id_f = open(os.path.join(coco_path, "annotations", "images2ids_{}.json".format(task)), "r")
        images_ids = json.load(id_f)

        if save_txt:
            dst_txt_path = os.path.join(save_path, "txt-{:.3f}".format(acc))
            self.create_dirs(dst_txt_path)
        if save_img:
            dst_img_path = os.path.join(save_path, "img-{:.3f}".format(acc))
            self.create_dirs(dst_img_path)

        for imgid, det in dt_info.items():
            if len(det["boxes"]):
                img_name = list(images_ids.keys())[list(images_ids.values()).index(imgid)]

                img_data = cv2.imread(os.path.join(img_path, img_name))
                h, w = img_data.shape[:2]

                if save_txt:
                    txt_file = open(os.path.join(dst_txt_path, img_name.replace("png", "txt")), "w", encoding="utf-8")
                    for cord in det["boxes"]:
                        bw = cord[2] - cord[0]
                        bh = cord[3] - cord[1]
                        cord = self.convert(size=[w, h], box=[cord[0], cord[1], bw, bh])
                        txt_file.write(' 0 {0} {1} {2} {3}\n'.format(cord[0], cord[1], cord[2], cord[3]))

                if save_img:
                    for cord in det["boxes"]:
                        cord[0], cord[1], cord[2], cord[3] = int(cord[0]), int(cord[1]), int(cord[2]), int(cord[3])

                        cv2.rectangle(img_data, (cord[0], cord[1]), (cord[2], cord[3]), color=(255, 255, 0),
                                      thickness=2)

                        cv2.imwrite(os.path.join(dst_img_path, img_name), img_data)

    @staticmethod
    def create_dirs(path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)


if __name__ == "__main__":
    task = "test"
    min_conf, max_conf = 0.1, 0.6
    conf_thres_list = np.linspace(min_conf, max_conf, int(np.round((max_conf - min_conf) / 0.001)) + 1, endpoint=True)
    # conf_thres_list = [0.412]
    pred_path = r"/home/mmdetection-2.16.0/resultscurdcnv2-json/zt_json"
    # pred_path = r"/home/projects/yolov5-5.0/infer/thyroid/exp1_v5s6"

    gt_path = r"/data/guofeng/detection/Thyroid/testData/{}Images".format(task)
    save_path = os.path.join(os.path.split(pred_path)[0], pred_path.split("/")[-1].split("_")[0] + "-metric")

    cocoDt_file = glob.glob(os.path.join(pred_path, '*.json'))[0]
    cocoGt_file = os.path.join(gt_path, "annotations", '{}Nodule.json'.format(task))

    metrics = Metrics(cocoGt_file=cocoGt_file,
                      cocoDt_file=cocoDt_file,
                      conf_thres_list=conf_thres_list)

    data_info = metrics.compute_nodule_acc(iou_thres=0.3)

    cocoGt_file = os.path.join(gt_path, "annotations", '{}.json'.format(task))

    video_paths = "/data/guofeng/detection/Thyroid/testData/{}Videos/".format(task)
    extractNode = extractNode(video_paths)

    metrics.create_dirs(save_path)
    for acc, info in data_info.items():
        metrics.save_det_results(info["dt_info"], save_path=save_path, save_txt=True,
                                 save_img=False, acc=acc, coco_path=gt_path, task=task)

        metrics.cocoGt_file = cocoGt_file
        metrics.cocoDt_file = cocoDt_file
        metrics.conf_thres_list = [info["conf"]]
        metrics.compute_nodule_f1()

        extractNode.txt_dirs = os.path.join(save_path, "txt-{:.3f}".format(acc))
        extractNode.output_dir = os.path.join(save_path, "node-{:.3f}".format(acc))
        metrics.create_dirs(extractNode.output_dir)

        max_fp = extractNode.result()

        max_tp = info["tp"]
        nodep = max_tp / (max_tp + max_fp + 1e-16)
        noder = acc

        nf1 = 2 * nodep * noder / (nodep + noder + 1e-16)
        print("Nodule level metric: conf:{:.3f}, FP:{}, precision:{:.3f}, recall:{:.3f}, F1:{:.3f}, Absence nodule:{}".format(
            info["conf"], max_fp, nodep, noder, nf1, info["node"]))

