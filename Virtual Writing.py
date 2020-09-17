import numpy as np
import cv2

vid = cv2.VideoCapture(0)
frameWidth = 440
frameHeight = 480
vid.set(3, frameWidth)
vid.set(4, frameHeight)
vid.set(10, 160)

colors = [[5, 107, 0, 19, 255, 255],   # Orange
          [133, 56, 0, 159, 156, 255],   # Purple
          [57, 76, 0, 100, 255, 255]]   # Green

colorValues = [[51, 153, 255],                       ## These values should be BGR format
               [255,0,155],
               [0, 255, 0]]

points = []   # [x, y, colorID]

def empty(a):
    pass

"""cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Bright Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Bright Max", "TrackBars", 255, 255, empty)"""

def findColor(imgResult):
    imgHSV = cv2.cvtColor(imgResult, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    for c in colors:
        lower = np.float32(c[0:3])
        upper = np.float32(c[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getCoutours(mask)
        cv2.circle(imgResult, (x,y), 10, colorValues[count],cv2.FILLED)
        if x!=0 and y!=0:
            newPoints.append([x, y, count])
        count += 1
        imgRes = cv2.bitwise_and(imgResult, imgResult, mask=mask)
        #cv2.imshow(str(c[0]), mask)
        #cv2.imshow("Video", img)
        cv2.imshow("result", imgRes)
    return newPoints

def getCoutours(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 30:
            cv2.drawContours(mask, cnt, -1, (255,0,0), 2)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x+w//2, y

def drawOnCanvas():
    for point in points:
        cv2.circle(imgResult, (point[0], point[1]), 10, colorValues[point[2]], cv2.FILLED)

while True:
    success, img = vid.read()
    imgResult = img.copy()
    newPoints = findColor(imgResult)
    if len(newPoints) != 0:
        for newPoint in newPoints:
            points.append(newPoint)
    if len(points) != 0:
        drawOnCanvas()
    cv2.imshow("Result", imgResult)
    if cv2.waitKey(1) == ord("q"):
        break