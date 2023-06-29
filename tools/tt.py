import requests
import pickle
from PIL import Image 
import cv2 

server_base = "http://127.0.0.1:5500"

img_path = "/home/walter/nas_cv/walter_stuff/od_video_skip_0/0020.jpg"
img = cv2.imread(img_path)


bbox = cv2.selectROI("test", img, False, False)
print(bbox)