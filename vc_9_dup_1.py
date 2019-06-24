import cv2, sys, numpy, os 
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 5)', '(5, 10)', '(10, 15)', '(15, 20)', '(20, 25)', '(25, 30)', '(36, 42)','(42,48)']
gender_list = ['Male', 'Female']
  
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
        face_img = im[y:y+h, h:h+w].copy()
		#DNN Stands for Deep Neural Network, It is used for preprocessing Images
		#It mainly works on Mean Subtraction, It calculates Average intesity of all pixels of each R G B Channel and subtract and do the Normalization
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        print(blob)
        age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
        gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        print("Gender : " + gender)
#Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        print("Age Range: " + age)
        overlay_text = "%s %s" % (gender, age)
  
        if prediction[1]<500: 
  
           cv2.putText(im, '% s - %.0f - Age :%s - Gender :%s' % 
(names[prediction[0]], prediction[1],age,gender), (x-10, y-10),  
cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
			
        else: 
          cv2.putText(im, 'not recognized',  
(x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 
        
	
  
        cv2.imshow('OpenCV', im) 
      
    key = cv2.waitKey(10) 
    if key == 27: 
        break