import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8m-pose.pt')
vid_path = 'ultralytics/assets/videos/p.mp4'

cap = cv2.VideoCapture(vid_path)
frame_index = 100

while cap.isOpened():
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    # frame_index = frame_index + 1
    success, frame = cap.read()
    if success:
        results = model(frame, conf=0.2)
        annotated_frame = results[0].plot()
        cv2.imshow("test", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break