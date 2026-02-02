import cv2 as cv

facemodel=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eyemodel=cv.CascadeClassifier("haarcascade_eye.xml")
webcam=cv.VideoCapture(0)
maskimage=cv.imread("face-mask.png")

while True:
    bool,frame =webcam.read()
    cv.imshow('mask_image',frame)
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    gr=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    face=facemodel.detectMultiScale(gray,1.3,5)
    eye=eyemodel.detectMultiScale(gr)

    for x,y,w,h in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv.imshow("Frame",frame)

    for x,y,w,h in eye:
        cv.rectangle(frame,(x,y),(x+w,y+h),(225,0,0),2)
    cv.imshow("Frame",frame)

    
    if cv.waitKey(1)== 27:
        break
webcam.release()
cv.destroyAllWindows()
