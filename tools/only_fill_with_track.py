from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

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


frames_dir = "/home/walter/nas_cv/walter_stuff/git/pysot/data/images/076150982312/cam_5"
frames = glob.glob(f"{frames_dir}/*.jpeg")
frames = sorted(frames)

first_frame = True
green = (0, 255, 0)
thickness = 2
cv2.namedWindow("video", cv2.WND_PROP_FULLSCREEN)

tracker_init = False 

for frame in frames:
    # print(frame)
    img = cv2.imread(frame)
    bbox = []
    results = query_yolov5(frame)
    if len(results) == 1:
        bbox, confidence, _, _ = parse_result_json(results[0])
        tracker = start_tracker()
        tracker.init(img, bbox)
        tracker_init = True
        print(confidence)
    elif tracker_init:
        outputs = tracker.track(img)
        bbox = outputs['bbox']
        print(outputs)
    
    if bbox:
        bbox = list(map(int, bbox))
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), green, thickness)

    cv2.imshow("video", img)
    cv2.waitKey(40)

