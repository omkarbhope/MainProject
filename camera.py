import cv2
from model import FacialExpressionModel
import dlib
import numpy as np

facec = cv2.CascadeClassifier('E://Youtube/Real-Time-Face-Expression-Recognition/haarcascade_frontalface_default.xml')
model = FacialExpressionModel("E://Youtube/Real-Time-Face-Expression-Recognition/model.json", "E://Youtube/Real-Time-Face-Expression-Recognition/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C://Users/omkar/Downloads/shape_predictor_68_face_landmarks.dat")

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self,red,blue,green,pigment=0.3):
        _, fr = self.video.read()
        output = fr.copy()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)
        faces2 = detector(gray_fr)

        def l(i):
            return (landmarks.part(i).x,landmarks.part(i).y)

        for face in faces2:

            landmarks = predictor(gray_fr, face)
            
            for n in range(0, 68):
                pts_array = np.array([l(54),l(55),l(56),l(57),l(58),l(59),l(48),l(60),l(67),l(66),l(65),l(64)],np.int32)
                pts_array = pts_array.reshape((-1,1,2))
                cv2.fillPoly(fr,[pts_array],(blue,green,red))
                pts_array2 = np.array([l(48),l(49),l(50),l(51),l(52),l(53),l(54),l(64),l(63),l(62),l(61),l(60)],np.int32)
                pts_array2 = pts_array2.reshape((-1,1,2))
                cv2.fillPoly(fr,[pts_array2],(blue,green,red))

        # for (x, y, w, h) in faces:
        #     fc = gray_fr[y:y+h, x:x+w]

        #     roi = cv2.resize(fc, (48, 48))
        #     pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

        #     cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
        #     cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.addWeighted(fr, pigment, output, 1 - 0.3,0, output)

        _, jpeg = cv2.imencode('.jpg', output)
        return jpeg.tobytes()
