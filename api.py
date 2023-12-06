from flask import Flask, jsonify, request, Response
import cv2
from ultralytics import YOLO
import numpy as np
from skimage.transform import resize
from util.dis_hm import get_row_col, draw_grid, distribution_heatmap
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/dis_hm', methods=['POST'])
def get_dis_hm():
    img = distribution_heatmap()
    response = Response(img.tobytes(), content_type='image/jpeg')
    return response  

@app.route('/api/pose_hm', methods=['POST'])
def get_pose_hm():
    img = distribution_heatmap()
    response = Response(img.tobytes(), content_type='image/jpeg')
    return response   

if __name__ == '__main__':
    app.run(debug=True, port = 5173)