# import cv2 as cv

# webcam=cv.VideoCapture(0)

# while True:
#     bool,frame =webcam.read()
#     cv.imshow('FRAME',frame)
#     if cv.waitKey(1)== ord('z'):
#         break
# webcam.release()
# cv.destroyAllWindows()

# rectange =cv.
#--------------------------------------------------------------------

import cv2 as cv
import numpy as np

blank=np.zeros((400,400),dtype='uint8')

rectangle=cv.rectangle(blank.copy(), (30,30),(370,370),255,-1)
circle=cv.circle(blank.copy(), (200,200),200,255,-1)

# cv.imshow('Rectangle',rectangle)
# cv.imshow('Circle',circle)

bitwise_and =cv.bitwise_and(rectangle,circle)
cv.imshow('Bitwise AND',bitwise_and)


bitwise_xor =cv.bitwise_xor(rectangle,circle)
cv.imshow('Xor',bitwise_xor)

cv.waitKey(0)