import torch
import cv2 
from PIL import Image

# Model
micro = "/home/walter/nas_cv/walter_stuff/git/yolov5-master/micro_controller/micro_controller/weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', micro)

# Image
im = '/home/walter/nas_cv/walter_stuff/od_video_skip_0/0019.jpg'
img = Image.open(im)

# Inference
results = model(img)

df = results.pandas().xyxy[0]
json_data = df.to_json(orient='records')

print(json_data)
# for index, row in df.iterrows():
#     print(index)
#     print(row['xmin'])
#     print(row['ymin'])
#     print(row['xmax'])
#     print(row['ymax'])
#     print(row['confidence'])
#     print(row['class'])

