import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C://Users/omkar/Downloads/shape_predictor_68_face_landmarks.dat")


def l(i):
    return (landmarks.part(i).x,landmarks.part(i).y)

while True:
    _, frame = cap.read()
    output = frame.copy()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    
    for face in faces:

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            # x = landmarks.part(n).x
            # y = landmarks.part(n).y
            # if n!=0:
            #     m=n-1
            #     x_prev = landmarks.part(m).x
            #     y_prev = landmarks.part(m).y
            pts_array = np.array([l(54),l(55),l(56),l(57),l(58),l(59),l(48),l(60),l(67),l(66),l(65),l(64)],np.int32)
            pts_array = pts_array.reshape((-1,1,2))
            cv2.fillPoly(frame,[pts_array],(0,0,255))
            pts_array2 = np.array([l(48),l(49),l(50),l(51),l(52),l(53),l(54),l(64),l(63),l(62),l(61),l(60)],np.int32)
            pts_array2 = pts_array2.reshape((-1,1,2))
            cv2.fillPoly(frame,[pts_array2],(0,0,255))

            
            # x_temp1= landmarks.part(48).x
            # y_temp1= landmarks.part(48).y
            # x_temp2= landmarks.part(59).x
            # y_temp2= landmarks.part(59).y

            # x_temp3= landmarks.part(60).x
            # y_temp3= landmarks.part(60).y
            # x_temp4= landmarks.part(67).x
            # y_temp4= landmarks.part(67).y

            # if n>48 and n<=59:
            #     cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            #     cv2.line(frame, (x_prev,y_prev), (x,y), (0, 0, 255), 1)
            # elif n>60 and n<=67:
            #     cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            #     cv2.line(frame, (x_prev,y_prev), (x,y), (0, 0, 255), 1)
            # # else:
            # #     cv2.circle(frame, (x, y), 2, (0, 0, 0), -1)
            
            # cv2.line(frame, (x_temp2,y_temp2), (x_temp1,y_temp1), (0, 0, 255), 1)
            # cv2.line(frame, (x_temp4,y_temp4), (x_temp3,y_temp3), (0, 0, 255), 1)

    cv2.addWeighted(frame, 0.3, output, 1 - 0.3,0, output)
    cv2.imshow("Frame", output)

    key = cv2.waitKey(1)
    if key == 27:
        break


