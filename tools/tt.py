import requests
import pickle
from PIL import Image 
import cv2 

img = cv2.imread("/home/walter/git/pysot/data/OB_walter/012993112059/100/00_0001.jpg")
region = cv2.selectROI("frame", img, False, False)
print(region)

region = [35, 35, 245, 245]