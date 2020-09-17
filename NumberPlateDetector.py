import numpy
import cv2

vid = cv2.VideoCapture(0)
vid.set(3, 640)
vid.set(4, 480)
vid.set(10, 150)

NPcascade = cv2.CascadeClassifier("haarcascades/haarcascade_russian_plate_number.xml")

while True:
    success, img = vid.read()

    NumPlates = NPcascade.detectMultiScale(img, 1.1, 4)

    for x, y, w, h in NumPlates:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(img, "Number Plate", (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Img", img)
    if cv2.waitKey(1) == ord("q"):
        break