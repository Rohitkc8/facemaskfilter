import cv2 as cv 
import numpy as np

web=cv.VideoCapture(0)
eyemodel=cv.CascadeClassifier("haarcascade_eye.xml")

glass=cv.imread("glassimg.png")

while True:
    bool,frame=web.read()

    grey=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    eye=eyemodel.detectMultiScale(grey,1.3,5)

    for x,y,w,h in eye:

        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    cv.imshow("Frame:",frame)
    
    if cv.waitKey(1)==27:
        break
web.release()
cv.destroyAllWindows()





