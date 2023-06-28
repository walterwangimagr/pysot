from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import re

import cv2
import torch
import numpy as np
import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from PIL import Image, ImageDraw
import requests
import pickle
import json
import time 


def query_yolov5(img_path):
    server_base = "http://127.0.0.1:5500"
    img = Image.open(img_path)
    headers = {'Content-Type': 'application/octet-stream'}
    response = requests.post(f"{server_base}/infer",
                             data=pickle.dumps(img), headers=headers)
    results = response.json()
    return json.loads(results['results'])


def parse_result_json(result):
    xmin = result['xmin']
    ymin = result['ymin']
    xmax = result['xmax']
    ymax = result['ymax']
    confidence = result['confidence']
    class_id = result['class']
    class_name = result['name']
    xyxy_bbox = [xmin, ymin, xmax, ymax]
    return xyxy_bbox, confidence, class_id, class_name


def start_tracker():
    config_file = "/home/walter/nas_cv/walter_stuff/git/pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml"
    cfg.merge_from_file(config_file)
    model_path = "/home/walter/nas_cv/walter_stuff/git/pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth"
    device = torch.device('cuda')
    model = ModelBuilder()
    model.load_state_dict(torch.load(model_path,
            map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)
    tracker = build_tracker(model)
    return tracker


def xyxy_to_xywh(xyxy_bbox):
    """
    xmin ymin xmax ymax to xmin ymin w h
    """
    xmin = xyxy_bbox[0]
    ymin = xyxy_bbox[1]
    xmax = xyxy_bbox[2]
    ymax = xyxy_bbox[3]
    w = xmax - xmin
    h = ymax - ymin 
    return [xmin, ymin, w, h]


def xywh_to_xyxy(xywh_bbox):
    """
    xmin ymin w h to xmin ymin xmax ymax 
    """
    xmin = xywh_bbox[0]
    ymin = xywh_bbox[1]
    w = xywh_bbox[2]
    h = xywh_bbox[3]
    xmax = xmin + w
    ymax = ymin + h
    return [xmin, ymin, xmax, ymax]

# write yolo label label, xywh normalized form 
def save_labels(save_dir, img_name, cls_id, normalized_bbox, confidence):
    """
    save xyxy bbox to xywh yolo format
    """
    os.makedirs(save_dir, exist_ok=True)
    label_name = re.sub(".jpeg", ".txt", img_name)
    savePath = os.path.join(save_dir, label_name)
    normalized_bbox = xyxy_to_xywh(normalized_bbox)
    
    with open(savePath, 'w') as f:
        str_to_save = f"{cls_id} {normalized_bbox[0]} {normalized_bbox[1]} {normalized_bbox[2]} {normalized_bbox[3]} {confidence}"
        f.write(str_to_save)

# read bbox from label file instead of run inf 
img_path = "/home/walter/nas_cv/walter_stuff/git/pysot/data/images/076150982312/cam_0/076150982312_0_cam_0_00003.jpeg"
def read_img_yolo_label(img_path):
    label_path = re.sub("images", "auto-labels-yolov5", img_path)
    label_path = re.sub(".jpeg", ".txt", label_path)
    with open(label_path, "r") as f:
        content = f.readline().split()
        cls_id = content[0]
        xmin = content[1]
        ymin = content[2]
        w = content[3]
        h = content[4]
        confidence = content[5]
    
    bbox = [xmin, ymin, w, h]
    bbox = xywh_to_xyxy(list(map(float,bbox)))
    return cls_id, bbox, confidence

print(read_img_yolo_label(img_path))
# for each folder, each camera 


# frames_dir = "/home/walter/nas_cv/walter_stuff/git/pysot/data/images/076150982312/cam_5"
# frames = glob.glob(f"{frames_dir}/*.jpeg")
# frames = sorted(frames)

# first_frame = True
# green = (0, 255, 0)
# thickness = 2
# cv2.namedWindow("video", cv2.WND_PROP_FULLSCREEN)

# tracker_init = False 

# for frame in frames:
#     # print(frame)
#     img = cv2.imread(frame)
#     bbox = []
#     results = query_yolov5(frame)
#     if len(results) == 1:
#         bbox, confidence, _, _ = parse_result_json(results[0])
#         tracker = start_tracker()
#         tracker.init(img, bbox)
#         tracker_init = True
#     elif tracker_init:
#         outputs = tracker.track(img)
#         bbox = outputs['bbox']
    
#     if bbox:
#         bbox = list(map(int, bbox))
#         cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), green, thickness)

#     cv2.imshow("video", img)
#     cv2.waitKey(40)

