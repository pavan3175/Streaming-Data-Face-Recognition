import os
from imageai.Detection import ObjectDetection

#It Works on ImageAi Library.

execution_path = 'C:/Users/pparepal/Documents/vc_python'

#Here We have ObjectDetection Class and we are creating Instance of a class.

detector = ObjectDetection()

#Here You will Perform Object Detection Using a pre-trained Model called Retinanet
detector.setModelTypeAsRetinaNet()
#Here we are pointing to a model which is pre-trained and loading here.
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

#The Detect Object From Images is used to detect Object from Images
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "man_1.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )