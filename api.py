from flask import Flask, jsonify, request, Response
import cv2
from ultralytics import YOLO
import numpy as np
from skimage.transform import resize
from util.dis_hm import get_row_col, draw_grid, distribution_heatmap


app = Flask(__name__)
@app.route('/api/dis_hm', methods=['GET'])
def get_dis_hm():
    return Response(distribution_heatmap(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/pose_hm', methods=['GET'])
def get_pose_hm():
    return Response(distribution_heatmap(), mimetype='multipart/x-mixed-replace; boundary=frame')    

if __name__ == '__main__':
    app.run(debug=True)