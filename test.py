import cv2
from ultralytics import YOLO
import numpy as np
from skimage.transform import resize

model = YOLO('yolo-overhead-person.pt')

vid_path = 'ultralytics/assets/videos/32_1_crop.mp4'

cap = cv2.VideoCapture(vid_path)
cap.set(cv2.CAP_PROP_FPS, 1)

first_im = cap.read()[1]
vid_shape = first_im.shape
width, height = vid_shape[1], vid_shape[0]
cell_size = 20
n_cols = int(width/cell_size)   
n_rows = int(height/cell_size) 
alpha = 0.4

heat_matrix = np.zeros((n_rows, n_cols, 3))
frame_num = 0

fps = (cap.get(cv2.CAP_PROP_FPS))

def get_row_col(x, y):
    row = int(y/cell_size)
    col = int(x/cell_size)
    return row, col

def draw_grid(image):
    for i in range(n_rows):
        start_point = (0, (i+1)*cell_size)
        end_point = (width, (i+1)*cell_size)
        color = (255,255,255)
        thickness = 1
        image = cv2.line(image, start_point,end_point,color,thickness)

    for i in range(n_cols):
        start_point = ((i+1)*cell_size, 0)
        end_point = ((i+1)*cell_size, height)
        color = (255,255,255)
        thickness = 1
        image = cv2.line(image, start_point,end_point,color,thickness)

    return image

while cap.isOpened():
    for i in range(0,60): cap.read()
    success, frame = cap.read()
    if success:
        results = model(frame, conf=0.2)
        boxes = results[0].boxes.xyxy.squeeze()
        item = boxes
        if boxes.dim() == 1:
            x1,y1,x2,y2 = int(item[0].item()),int(item[1].item()),int(item[2].item()),int(item[3].item())
            p1, p2 = get_row_col(x1,y1), get_row_col(x2,y2)
            for i in range(p1[0], p2[0]):
                for j in range(p1[1], p2[1]):
                    heat_matrix[i][j]+=1
                        

            # temp_heat = heat_matrix.copy()
            # temp_heat = resize(temp_heat, (height,width))
            # temp_heat = temp_heat/np.max(temp_heat)
            # temp_heat = np.uint8(temp_heat*255)

            # image_heat = cv2.applyColorMap(temp_heat, cv2.COLORMAP_JET)
            # cv2.addWeighted(image_heat, alpha, frame, 1-alpha, 0, frame)

            # cv2.imshow("test", frame)
            # # cv2.imshow("test", image_heat)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break
    else:
        break

temp_heat = heat_matrix.copy()
temp_heat = resize(temp_heat, (height,width))
temp_heat = temp_heat/np.max(temp_heat)
temp_heat = np.uint8(temp_heat*255)

image_heat = cv2.applyColorMap(temp_heat, cv2.COLORMAP_JET)
cv2.addWeighted(image_heat, alpha, first_im, 1-alpha, 0, first_im)
cv2.imshow('test',first_im)
cv2.waitKey()