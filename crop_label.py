import cv2
from ultralytics import YOLO
import os


model = YOLO('yolo-overhead-person.pt')

# vid_path = 'ultralytics/assets/videos/37_1_crop.mp4'

# cap = cv2.VideoCapture(vid_path)

# while cap.isOpened():
#     success, frame = cap.read()
#     if success:
#         results = model(frame)
#         annotated_frame = results[0].plot()
#         cv2.imshow("test", annotated_frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         break


def func():
    im_path = 'D:/download/pose-estimation.v1i.clip/valid/stand'
    des_path = 'D:/download/pose-estimation.v1i.clip/valid/newstand'
    img_num = 1 
    for nam in range(0,1):
        for filename in os.listdir(im_path):
            img = cv2.imread(os.path.join(im_path,filename))
            results = model(img)
            boxes = results[0].boxes.xyxy.squeeze()
            detect_list = []
            #if detect only 1 
            if boxes.dim() == 1:
                item = boxes
                detect_list.append([int(item[0].item()),int(item[1].item()),int(item[2].item()),int(item[3].item())])
            ##more than 1 
            else:
                for item in boxes:
                    detect_list.append([int(item[0].item()),int(item[1].item()),int(item[2].item()),int(item[3].item())])

            for index, sub_xyxy in enumerate(detect_list):
                    x1,y1,x2,y2 = sub_xyxy[0],sub_xyxy[1],sub_xyxy[2],sub_xyxy[3]
                    sub_im = img[y1:y2,x1:x2]
                    cv2.imwrite(os.path.join(des_path,'{ind}.jpg'.format(ind=img_num)),sub_im)
                    img_num+=1
list = ['high','stand','medium-kneel','low-kneel']
func()

