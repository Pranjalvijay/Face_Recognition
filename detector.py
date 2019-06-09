import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer\\trainingData.yml")
Id = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if (conf>50):
            if (Id == 1):
                Id="Pranjal"
            else:
                Id="Unknown"
        
        cv2.putText(img, str(Id),(x,y+h), font, 2, 255)
        #print(x,y,w,h)

    cv2.imshow("Face", img)

    if(cv2.waitKey(1) == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
