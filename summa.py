import cv2 as cv

img=cv.imread("face-mask.png")

cv.imshow("img",img)
cv.waitKey()