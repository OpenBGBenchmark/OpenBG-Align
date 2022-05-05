import io
import base64
import argparse
from tqdm import tqdm
import json
from PIL import Image
import csv
import os
os.system("nvcc -V")

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torch

print("torch")
print(torch.__version__)
print(torch.cuda.is_available())

##############################################
import numpy as np
import cv2
import torch

NUM_OBJECTS = 36

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, \
    fast_rcnn_inference_single_image

FIELDNAMES = ['item_id', 'image_h', 'image_w', 'num_boxes', 'boxes', 'features', 'cls_prob', 'title']

MIN_BOXES = 36
MAX_BOXES = 36


##############################################
def get_detections_from_image(predictor, raw_image):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        # print("original image size: ",raw_height,raw_width)

        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        # image = predictor.aug.get_transform(raw_image).apply_image(raw_image)
        # print("Transformed image size: ", image.shape[:2])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]
        # print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape) # 154 x 4

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        # print('Pooled features size:', feature_pooled.shape) # 154 x 2048

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        # print("outputs: ",outputs)
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]

        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:],
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            )
            if len(ids) == NUM_OBJECTS:  # 36个框
                break
        # 由800 x 800 换成 736 x 736
        instances = detector_postprocess(instances, raw_height, raw_width)  # 恢复出原来的图片大小
        roi_features = feature_pooled[ids].detach()  # 36 x 2048

        selected_probs = probs[ids]
        # print("selected_probs: ",selected_probs.size())

        if torch.sum(torch.isnan(roi_features)) > 0:
            return None

        return_data = {
            "image_h": raw_height,
            "image_w": raw_width,
            "num_boxes": len(ids),
            "boxes": json.dumps(instances.pred_boxes.tensor.cpu().numpy().tolist()),
            "features": json.dumps(roi_features.cpu().numpy().tolist())
        }

    return return_data


def get_predictor(args):
    print("get_predictor")
    cfg = get_cfg()

    # define myself
    cfg.MODEL.PROPOSAL_GENERATOR.HID_CHANNELS=512
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"

    cfg.DATASETS.TRAIN=("coco_2017_train",)
    cfg.DATASETS.TEST=("coco_2017_val",)

    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.STEPS = (60000, 80000)
    cfg.SOLVER.MAX_ITER = 90000

    cfg.INPUT.MIN_SIZE_TRAIN=(640, 672, 704, 736, 768, 800)


    cfg.MODEL.MASK_ON=False
    cfg.MODEL.RESNETS.DEPTH = 101
    cfg.MODEL.PROPOSAL_GENERATOR.HID_CHANNELS = 512
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64, 128, 256, 512]]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1600
    cfg.MODEL.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
    cfg.MODEL.ROI_BOX_HEAD.RES5HALVE = False
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIPool"
    cfg.SOLVER.STEPS = (210000, 250000)
    cfg.MODEL.MAX_ITER = 270000

    # cfg.merge_from_file("configs/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml")
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # VG Weight
    # cfg.MODEL.WEIGHTS = "https://gongxiangdm.oss-cn-zhangjiakou.aliyuncs.com/alin/multi_modal/bp_models/faster_rcnn_from_caffe_attr.pkl"
    cfg.MODEL.WEIGHTS = args.save_model_path

    predictor = DefaultPredictor(cfg)
    print("predictor: ", predictor)
    return predictor

import signal
def set_timeout(num, callback):
    def wrap(func):
        def handle(signum, frame):
            raise RuntimeError

        def to_do(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)
                signal.alarm(num)
                r = func(*args, **kwargs)
                signal.alarm(0)
                return r
            except RuntimeError as e:
                callback()

        return to_do

    return wrap
def after_timeout():
    print("Time out!")

@set_timeout(50,after_timeout)
def doit_xl(args, main_pict, predictor):
    # 读取本地图片
    image = Image.open(open(os.path.join(args.local_image_path, main_pict), mode='rb'))
    if args.convertL == "L":
        image = image.convert("L")

    image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(np.asarray(image_bgr), cv2.COLOR_BGR2RGB)
    if type(image) == type(None) or type(image_bgr) == type(None):
        return None
    detection_feature = get_detections_from_image(predictor, image_rgb)

    return detection_feature


def get_bp_feature(args, table_data,predictor,writer):
    res_records=[]
    for index,record in tqdm(enumerate(table_data)):
        item_id,title,main_pict=record
        try:
            detection_feature=doit_xl(args, main_pict, predictor)
            if detection_feature == None:
                continue

            res_records.append({"item_id": item_id,
                                "image_h": detection_feature["image_h"],
                                "image_w": detection_feature["image_w"],
                                "num_boxes": detection_feature["num_boxes"],
                                "boxes": detection_feature["boxes"],
                                "features": detection_feature["features"],
                                "title": title
                                })

        except Exception as e:
            print("image_file: ", main_pict, " item_id: ", item_id)
            print("error: ", e)
            pass

        if index%1000==0:
            print("write")
            writer.writerows(res_records)
            res_records=[]

    print("write")
    writer.writerows(res_records)

    return

##############################################


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch table IO')
    parser.add_argument('--envs', default="local")
    # local
    parser.add_argument('--input_file', default="", type=str, help="输入info文件，json")
    parser.add_argument('--local_image_path', default="", type=str, help="相关的图片路径")
    parser.add_argument('--output_file', default="", type=str, help="输出主体embedding结果")
    parser.add_argument('--save_model_path', default="", type=str, help="目标检测模型路径")
    parser.add_argument("--convertL", type=str, default="RGB")
    parser.add_argument("--thread_num", default=16, type=int)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args=get_args()
    predictor=get_predictor(args)
    table_data = []
    with open(args.input_file, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            item_id = line['item_id']
            title = line['title']
            item_image_name = line['item_image_name']
            table_data.append((item_id, title, item_image_name))
    tsvfile = open(args.output_file, 'w', encoding='utf-8')
    writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
    get_bp_feature(args,table_data,predictor,writer)
    tsvfile.close()














