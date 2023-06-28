import requests
import pickle
from PIL import Image 

server_base = "http://127.0.0.1:5500"

img_path = "/home/walter/nas_cv/walter_stuff/od_video_skip_0/0020.jpg"
img = Image.open(img_path)


headers = {'Content-Type': 'application/octet-stream'}
response = requests.post(f"{server_base}/infer", data=pickle.dumps(img), headers=headers)
results = response.json()

print(results)