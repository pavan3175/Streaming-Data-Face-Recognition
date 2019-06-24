import numpy
from imageai.Detection import VideoObjectDetection
import os

execution_path = 'C:/Users/pparepal/Documents/vc_python'
detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "video_2.mp4"),
                            output_file_path=os.path.join(execution_path, "traffic_detected")
                            , frames_per_second=20, log_progress=True)
print(video_path)