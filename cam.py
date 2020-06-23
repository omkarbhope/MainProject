import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import dlib
from model import FacialExpressionModel

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C://Users/omkar/Downloads/shape_predictor_68_face_landmarks.dat")

def emptyFunction(): 
	pass


def detect_face(img,red,blue,green):

    def l(i):
        return (landmarks.part(i).x,landmarks.part(i).y)
    
    face_img=img.copy()
    output = face_img.copy()

    gray_fr = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr, 1.3, 5)
    faces2 = detector(gray_fr)

    for face in faces2:

        landmarks = predictor(gray_fr, face)
	    
        for n in range(0, 68):
            pts_array = np.array([l(54),l(55),l(56),l(57),l(58),l(59),l(48),l(60),l(67),l(66),l(65),l(64)],np.int32)
            pts_array = pts_array.reshape((-1,1,2))
            cv2.fillPoly(face_img,[pts_array],(blue,green,red))
            pts_array2 = np.array([l(48),l(49),l(50),l(51),l(52),l(53),l(54),l(64),l(63),l(62),l(61),l(60)],np.int32)
            pts_array2 = pts_array2.reshape((-1,1,2))
            cv2.fillPoly(face_img,[pts_array2],(blue,green,red))
    
    for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(face_img, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.addWeighted(face_img, 0.3, output, 1 - 0.3,0, output)
        
    return (output)

cap=cv2.VideoCapture(0)
capture_duration = 100000
start_time = time.time()
predictions=[]
cv2.namedWindow("Face Detection Video")
cv2.createTrackbar('Blue', 'Face Detection Video', 0, 255, emptyFunction)
cv2.createTrackbar('Green', 'Face Detection Video', 0, 255, emptyFunction)
cv2.createTrackbar('Red', 'Face Detection Video', 0, 255, emptyFunction)
while (True):
    
    ret,face_img=cap.read(0)
    blue = cv2.getTrackbarPos("Blue", "Face Detection Video")
    green = cv2.getTrackbarPos("Green", "Face Detection Video")
    red = cv2.getTrackbarPos("Red", "Face Detection Video")
    face_img=detect_face(face_img,red,blue,green)
    # predictions.append(pred1)
    
    cv2.imshow('Face Detection Video',face_img)
    
    if cv2.waitKey(3) & 0xFF==27:
        break

def most_frequent( List ):
    max_prediction= max ( set ( List ), key = List .count) 
    return max_prediction

# print(most_frequent(predictions))
cap.release()
cv2.destroyAllWindows()



        