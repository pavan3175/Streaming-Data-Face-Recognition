import cv2, sys, numpy, os 
haar_file = 'haarcascade_frontalface_default.xml'
  
# All the faces data will be 
#  present this folder 
datasets = 'datasets'  
  
  
# These are sub data sets of folder,  
# for my faces I've used my name you can  
# change the label here 
sub_data = sys.argv[1]     
  
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