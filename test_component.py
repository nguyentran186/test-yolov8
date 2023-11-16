import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8m-pose.pt')
cls_model = YOLO('yolostandpose.pt')

vid_path = 'ultralytics/assets/videos/41_2_crop.mp4'

cap = cv2.VideoCapture(vid_path)


while cap.isOpened():
    for i in range(0,1): cap.read()
    success, frame = cap.read()
    if success:
        results = model(frame)
        boxes = results[0].boxes.xyxy.squeeze() 
        detect_list = []
        result_keypoint = results[0].keypoints.xyn.cpu().numpy()[0]

        #if detect only 1 
        if boxes.dim() == 1:
            item = boxes
            detect_list.append([int(item[0].item()),int(item[1].item()),int(item[2].item()),int(item[3].item())])
        ##more than 1 
        else:
            for item in boxes:
                detect_list.append([int(item[0].item()),int(item[1].item()),int(item[2].item()),int(item[3].item())])
        print(detect_list)

        
        lb_list=[]
        if len(detect_list)>0:
            for ind, item in enumerate(detect_list):
                rshoulder , lshoulder, rleg, lleg = result_keypoint[6],result_keypoint[5],result_keypoint[12],result_keypoint[11]
                diff = [rshoulder[1]-rleg[1], lshoulder[1]-lleg[1]]

                x1,y1,x2,y2 = item
                sub_im = frame[y1:y2,x1:x2]
                cls_res = cls_model(sub_im)
                probs = cls_res[0].probs.data
                # print(probs)

                if (x1 < 800 or y2 > 500):
                    #### Kneel
                    if rshoulder[1]<0.7 and lshoulder[1]<0.7 and \
                        rshoulder[0]<0.8 and lshoulder[0]<0.8 and\
                        (diff[0]<-0.07 or diff[1]<-0.07):
                        if (probs[1].item()>0.3):
                            lb_list.append('kneel')
                        else:
                            lb_list.append('kneel')
                    #### Stand
                    else:
                        if (probs[0].item()>0.5):
                            lb_list.append('high')
                        else:
                            lb_list.append('stand')
                else: lb_list.append('person')


 
        annotated_frame = results[0].cls_plot(label_list = lb_list)
        cv2.imshow("test", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# model = YOLO('yolov8n.pt')
# # model = YOLO('yolo-stand-100ep.pt')
# sklt_model = YOLO('yolov8n-pose.pt')

# im_path = 'ultralytics/assets/zidane.jpg'
# # im_path = 'ultralytics/cfg/datasets/pose/train/images/29_1_crop_mp4-44_jpg.rf.315383544e1c0623c2424b2b3c02b94d.jpg'

# im = cv2.imread(im_path)
# results = model(im, conf=0.4)
# print(results[0].boxes)
# boxes = results[0].boxes.xyxy.squeeze()
# detect_list = []

# #if detect only 1 
# if boxes.dim() == 1:
#     item = boxes
#     detect_list.append([int(item[0].item()),int(item[1].item()),int(item[2].item()),int(item[3].item())])
#     # print(detect_list)
# ##more than 1 
# else:
#     for item in boxes:
#         detect_list.append([int(item[0].item()),int(item[1].item()),int(item[2].item()),int(item[3].item())])
#     # print(detect_list)

# # annotated_frame = results[0].plot()
# # x1,y1,x2,y2 = detect_list[0][0],detect_list[0][1],detect_list[0][2],detect_list[0][3]
# # annotated_frame = im[y1:y2,x1:x2]
# # cv2.imshow("test", annotated_frame)
# # cv2.waitKey()

# for sub_xyxy in detect_list:
#     x1,y1,x2,y2 = sub_xyxy[0],sub_xyxy[1],sub_xyxy[2],sub_xyxy[3]
#     sub_im = im[y1:y2,x1:x2]
#     sub_result = sklt_model(sub_im)
    