import numpy
from imageai.Detection import VideoObjectDetection
import os
import cv2


execution_path = 'C:/Users/pparepal/Documents/vc_python'

c1=cv2.VideoCapture(0)
detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=c1,
                            output_file_path=os.path.join(execution_path, "traffic_detected_1")
                            , frames_per_second=20, log_progress=True,minimum_percentage_probability=30)
print(video_path)