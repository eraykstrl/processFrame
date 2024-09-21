import cv2
from ultralytics import YOLO
import numpy as np
class image():
    def __init__(self):
        self.model=YOLO("yolov8n.pt")

    def process_frame(self):
        cap=cv2.VideoCapture(0)

        while True:
            ret,frame=cap.read()
            results=self.model.track(frame,tracker="bytetrack.yaml",conf=0.5,iou=0.5)
            annotated_frame=frame.copy()
            yukseklik,genislik=annotated_frame.shape

            for r in results:
                if len(r) > 0:
                    annotated_frame=r.plot()
                    for kutu in r.boxes.xyxy.cpu().np():
                        x_center = int((kutu[0] + kutu[2]) / 2)
                        y_center = int((kutu[1] + kutu[3]) / 2)
                        print(x_center)
                        print(y_center)

                else:
                    print("there is no object now ")

img=image()
img.process_frame()
