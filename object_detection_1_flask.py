import os
from imageai.Detection import ObjectDetection
from flask import Flask
from flask_restful import Api,request




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#port = int(os.getenv("PORT"))
app = Flask(__name__)

# app.config['UPLOAD_FOLDER'] = r"C:\Users\Documents"
api = Api(app)

#It Works on ImageAi Library.
@app.route('/object_detection', methods=['GET', 'POST'])
def train():
    execution_path = 'C:/Users/pparepal/Documents/vc_python'
    
    #Here We have ObjectDetection Class and we are creating Instance of a class.
    
    detector = ObjectDetection()
    
    #Here You will Perform Object Detection Using a pre-trained Model called Retinanet
    detector.setModelTypeAsRetinaNet()
    #Here we are pointing to a model which is pre-trained and loading here.
    detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    image=request.files["File"]
    #The Detect Object From Images is used to detect Object from Images
    detections = detector.detectObjectsFromImage(input_image=image, output_image_path=os.path.join(execution_path , "imagenew.jpg"))
    return "Succesful"
	
	
#for eachObject in detections:
    #print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
	
if __name__ == '__main__':
    app.run(debug=True)