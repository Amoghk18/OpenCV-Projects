import numpy as np
import cv2

frameWidth = 640
frameHeight = 440
vid = cv2.VideoCapture(0)
vid.set(3, frameWidth)
vid.set(4, frameHeight)
vid.set(10, 150)
imgW = 640
imgH = 480


def preprocessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    ker = 6 * np.ones((5,5))
    imgDilation = cv2.dilate(imgCanny, ker, iterations=1)
    imgErode = cv2.erode(imgDilation, ker, iterations=1)
    return imgErode

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 700:
            #cv2.drawContours(imgCont, cnt, -1, (255,0,0), 2)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgCont, biggest, -1, (255, 0, 0), 20)
    return biggest

def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [imgW, 0], [0, imgH], [imgW, imgH]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOut = cv2.warpPerspective(img, matrix, (imgW, imgH))

    imgCropped = imgOut[20:imgOut.shape[0]-20, 20:imgOut.shape[1]-20]
    imgCropped = cv2.resize(imgCropped, (imgOut.shape[0], imgOut.shape[1]))

    return imgOut

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    newP = np.zeros((4,1,2), np.int32)
    add = myPoints.sum(1)
    newP[0] = myPoints[np.argmin(add)]
    newP[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    newP[1] = myPoints[np.argmin(diff)]
    newP[2] = myPoints[np.argmax(diff)]
    return newP

while True:
    success, img = vid.read()
    img = cv2.resize(img, (imgW, imgH))
    imgCont = img.copy()
    imgTh = preprocessing(img)
    biggest = getContours(imgTh)
    if biggest.size !=0:
        imgWarp = getWarp(img, biggest)
    else:
        imgWarp = img.copy()
    cv2.imshow("Img", imgWarp)
    if cv2.waitKey(1) == ord("q"):
        break
