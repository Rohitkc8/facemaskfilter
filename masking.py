# import cv2 as cv
# import numpy as np


# img = cv.imread('image1.jpg')
# resized = cv.resize(img, (600, 800))  
# cv.imshow('Resized', resized)

# himg=cv.imread('humanimg.png')
# hresized= cv.resize(himg,(700,900))

# cv.imshow('human_Rsized',hresized)
# # blank = np.zeros(resized.shape[:2], dtype='uint8')
# # cv.imshow("BLANK IMAGE", blank)

# # mask = cv.circle(blank, (resized.shape[1]//2, resized.shape[0]//2), 100, 255, -1)
# # cv.imshow("MASK", mask)

# # masked = cv.bitwise_and(resized, resized, mask=mask)
# # cv.imshow('MASK Image', masked)

# cv.waitKey(0)
# cv.destroyAllWindows()

import cv2

trainedfacemodel = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
maskimg = cv2.imread("face-mask.png")  
webcam = cv2.VideoCapture(0)

while True:
    bool, frame = webcam.read()

    faces = trainedfacemodel.detectMultiScale(frame)

    for (x, y, w, h) in faces:
        cap_w = w
        cap_h = int(h * 0.6)

        cap_resize = cv2.resize(maskimg, (cap_w, cap_h))

        y1 = y - cap_h
        y2 = y
        x1 = x
        x2 = x + cap_w

        if y1 < 0:
            continue

        frame[y1:y2, x1:x2] = cap_resize

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == 27 :
        break

webcam.release()
cv2.destroyAllWindows()
