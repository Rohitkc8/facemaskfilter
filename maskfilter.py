import cv2 as cv
face = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv.VideoCapture(0)
mask = cv.imread("face-mask.png")

while True:
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        x1 = x
        y1 = y + int(h * 0.55)
        x2 = x1 + w
        y2 = y1 + h

        if x2 > frame.shape[1] or y2 > frame.shape[0]:
            continue
    
        mask_resized = cv.resize(mask, (w, h))

        frame[y1:y2, x1:x2] = mask_resized

    cv.imshow("Mask", frame)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
