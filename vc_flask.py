import cv2, sys, numpy, os 
from flask import Flask
from flask_restful import Api,request




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#port = int(os.getenv("PORT"))
app = Flask(__name__)

# app.config['UPLOAD_FOLDER'] = r"C:\Users\Documents"
api = Api(app)

@app.route('/train', methods=['GET', 'POST'])
def train():
    haar_file = 'haarcascade_frontalface_default.xml'
      
    # All the faces data will be 
    #  present this folder 
    datasets = 'datasets'  
      
      
    # These are sub data sets of folder,  
    # for my faces I've used my name you can  
    # change the label here 
    sub_data = request.json['name']    
      
    #So here we are joining the folder and our sub_Folder name`
    path = os.path.join(datasets, sub_data) 
    if not os.path.isdir(path): 
        os.mkdir(path) 
      
    # defining the size of images  
    (width, height) = (130, 100)     
      
    #The model is Pre-trained how to detect faces,ears,eyes..e.t.c 
    #These are stored in .xml files and the below code snippet is used to load that pre-trained model
    #Using CV2.CASCADE we detect out faces.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
    
    #'0' is used for my webcam,  
    # if you've any other camera 
    #  attached use '1' like this 
    webcam = cv2.VideoCapture(0)  
      
    # The program loops until it has 30 images of the face. 
    count = 1
    while count < 50:  
        (_, im) = webcam.read() 
    	#It Converts  Image from BGR to Gray
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    	#It will identify our face in our frame.
        faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
        for (x, y, w, h) in faces: 
    		#Here we are drawing a rectangle on our frame at x,y position and the bottompoints are detected by x+w,y+h (255,0,0):color,thickness:2
            cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2) 
    		#
            face = gray[y:y + h, x:x + w] 
            face_resize = cv2.resize(face, (width, height)) 
            cv2.imwrite('% s/% s.png' % (path, count), face_resize) 
        count += 1
          
        cv2.imshow('OpenCV', im) 
        key = cv2.waitKey(20) 
        if key == 27: 
            break
    return "Succesful"

@app.route('/predict', methods=['GET', 'POST'])	
def predict():
    haar_file = 'haarcascade_frontalface_default.xml'
    datasets = 'datasets'
      
    # Part 1: Create fisherRecognizer 
    print('Recognizing Face Please Be in sufficient Lights...') 
      
    # Create a list of images and a list of corresponding names 
    (images, lables, names, id) = ([], [], {}, 0) 
    for (subdirs, dirs, files) in os.walk(datasets): 
        for subdir in dirs: 
            names[id] = subdir 
            subjectpath = os.path.join(datasets, subdir) 
            for filename in os.listdir(subjectpath): 
                path = subjectpath + '/' + filename 
                lable = id
                images.append(cv2.imread(path, 0)) 
                lables.append(int(lable)) 
            id += 1
    (width, height) = (130, 100) 
      
    # Create a Numpy array from the two lists above 
    (images, lables) = [numpy.array(lis) for lis in [images, lables]] 
      
    # OpenCV trains a model from the images 
    # NOTE FOR OpenCV2: remove '.face' 
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, lables) 
      
    # Part 2: Use fisherRecognizer on camera stream 
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    webcam = cv2.VideoCapture(0) 
    while True: 
        (_, im) = webcam.read() 
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
        for (x, y, w, h) in faces: 
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
            face = gray[y:y + h, x:x + w] 
            face_resize = cv2.resize(face, (width, height)) 
            # Try to recognize the face 
            prediction = model.predict(face_resize) 
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3) 
      
            if prediction[1]<500: 
      
               cv2.putText(im, '% s - %.0f' % 
    (names[prediction[0]], prediction[1]), (x-10, y-10),  
    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 
            else: 
              cv2.putText(im, 'not recognized',  
    (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 
    	
      
        cv2.imshow('OpenCV', im) 
          
        key = cv2.waitKey(10) 
        if key == 27: 
            break
        
if __name__ == '__main__':
    app.run(debug=True)