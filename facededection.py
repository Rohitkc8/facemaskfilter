import cv2 as cv
import numpy as np
facemodel=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
web=cv.VideoCapture(0)
while True:

    bool,frame=web.read()
    grayframe=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    face=facemodel.detectMultiScale(grayframe)
    
    for(x,y,w,h) in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    cv.imshow("FRAME:",frame)
    if cv.waitKey(1)==27:
        quit()



