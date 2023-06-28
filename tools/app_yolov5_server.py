import torch 
from flask import Flask, request, Response
import jsonpickle
import pickle

app = Flask(__name__)

micro = "/home/walter/nas_cv/walter_stuff/git/yolov5-master/micro_controller/micro_controller/weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', micro)


@app.route('/infer', methods=['POST'])
def infer():
    r = request
    img = pickle.loads(r.get_data())
    results = model(img)
    df = results.pandas().xyxy[0]
    json_data = df.to_json(orient='records')

    response = {'results': json_data}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")
    
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5500)