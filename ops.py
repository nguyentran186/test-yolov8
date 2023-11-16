import cv2
from ultralytics import YOLO
import numpy as np

# model = YOLO('yolo-overhead-person.pt')
# model = YOLO('yolo-stand-100ep.pt')
model = YOLO('yolov8n-pose.pt')

vid_path = 'ultralytics/assets/videos/41_2_crop.mp4'

cap = cv2.VideoCapture(vid_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)
        # boxes = results[0].boxes.xyxy.squeeze()
        # detect_list = []
        # #if detect only 1 
        # if boxes.dim() == 1:
        #     item = boxes
        #     detect_list.append([int(item[0].item()),int(item[1].item()),int(item[2].item()),int(item[3].item())])
        # ##more than 1 
        # else:
        #     for item in boxes:
        #         detect_list.append([int(item[0].item()),int(item[1].item()),int(item[2].item()),int(item[3].item())])

        # for index, sub_xyxy in enumerate(detect_list):
        #     x1,y1,x2,y2 = sub_xyxy[0],sub_xyxy[1],sub_xyxy[2],sub_xyxy[3]
        #     sub_im = frame[y1:y2,x1:x2]
        #     sub_result = sklt_model(sub_im)

        annotated_frame = results[0].plot()
        boxes = results[0].boxes.xyxy.squeeze()
        if boxes.dim() == 1:
            item = boxes
            x1,y1,x2,y2 = int(item[0].item()),int(item[1].item()),int(item[2].item()),int(item[3].item())
            annotated_frame = annotated_frame[y1:y2, x1:x2]

        cv2.imshow("test", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break



    