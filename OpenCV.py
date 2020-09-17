import cv2
import numpy as np

######################## Loading a image, Video and switching on webcam #############################

img_path = "C:/Users/amogh/OneDrive/Pictures/Camera Roll/ILC.jpg"
"""img_ILC = cv2.imread(path)
cv2.imshow("Image", img_ILC)
cv2.waitKey(0)

vid_path = "C:/Users/amogh/Videos/Captures/test.mp4"
cap = cv2.VideoCapture(vid_path)  # We can give argument as 0 to turn on web cam
cap.set(3, 640)                   # We can set the property 3, Id 3 is width of the frame
cap.set(4, 480)                   # Id 4 is height of the frame
cap.set(10, 100)                  # Id 10 isBrightness """

# Other Properties
"""0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
5. CV_CAP_PROP_FPS Frame rate.
6. CV_CAP_PROP_FOURCC 4-character code of codec.
7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)"""

"""while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) == ord("q"):  # cv2.waitKey(1) returns the ascii value of key pressed and ord()
        break                       # fuction returns unicode of "q" """


##################################### Manipulating Images ##############################################

"""
img_ILC = cv2.imread(img_path)
imgGray = cv2.cvtColor(img_ILC, cv2.COLOR_BGR2GRAY)   # There are amny types of color transformations
imgBlur = cv2.GaussianBlur(img_ILC, (7,7), 0)         # Higher the dimension of the kernel more blur
                                                      # There are other types of blur also
                                                      # It takes arguments Image, Kernel_Size(as in convNets)
                                                      # SigmaX is standard deviation of the kernel/filter
                                                      # The standard deviation is calculated row wise
                                                      # Higher the sigma more the blur

imgCanny = cv2.Canny(img_ILC, 100, 100)               # It is an edge Detector(Canny Edge Detector)
                                                      # It taes in parameters Image, Threshold 1,Threshold 2
                                                      # If a pixel gradient is higher than the upper
                                                      # threshold, the pixel is accepted as an edge.
                                                      # If a pixel gradient value is below the lower
                                                      # threshold, then it is rejected.

# Some times when detecting edges we may fail to do so as the edge may be deformed, so we use
# Image Dilation which thickens the edges, it should be performed only on canny images
Kernel = np.ones((5,5))
imgDilation = cv2.dilate(imgCanny, Kernel,iterations=1)      # It takes input the image to dilate,
                                                             # kernel/filter size, Also we have to give
                                                             # the values that will be present
                                                             # in the filter, iterations is the number of
                                                             # times it shouldthicken the edge

# We can also erode the images, It makes colored images similar to oil painted images
imgEroded = cv2.erode(imgDilation, Kernel, iterations=1)
imgerodeILC = cv2.erode(img_ILC, Kernel, iterations=2)

cv2.imshow("Oil painting ILC", imgerodeILC)
cv2.imshow("erorded", imgEroded)
cv2.imshow("Edge Detector", imgCanny)
cv2.imshow("Blur Image",imgBlur)
cv2.imshow("Gray", imgGray)
cv2.imshow("Dilation", imgDilation)
cv2.waitKey(0) """

##################################### Resizing and Cropping ########################################
"""
img_ILC = cv2.imread(img_path)
print(img_ILC.shape)

imgResized = cv2.resize(img_ILC, (400,300))          # It takes image, and the target size
                                                     # to resize (width, height), if we increase the size
                                                     # of the image, the Quality of Image doesn't increase

# IMPORTANT : While Cropping an image no function is needed we can so it by the method of slicing/indexing
# Also while indexing height, width should be mentioned in this manner => [height, width]
# Detailed version =>>  [ height_start : height_end, width_start : width_end ]
imgCrop = img_ILC[170:500,550:900]                   # coordinates of my face

cv2.imshow("original image", img_ILC)
cv2.imshow("resized image", imgResized)
cv2.imshow("cropped image", imgCrop)
cv2.waitKey(0) """

########################### Drawing shapes,coloring and writing Texts on images #################################
"""
img = np.zeros((400,512,3),np.uint8)

#img[200:300, 200:300] = 255,0,0 # This gives color to the image

# Drawing Lines on an Image
cv2.line(img, (0,0), (300,300), (0,255,255), 1)  # This take in parameter the image, starting point,
                                                 # ending point, color of the line and the thickness
                                                 # while defining starting and ending points it must be
                                                 # defined as (width, height) and img.shape is of the form
                                                 # (height, width, channels)

cv2.line(img, (0,10), (img.shape[1], img.shape[0]), (0,255,255), 1)

# Drawing rectangles
cv2.rectangle(img, (100,100), (200,200), (255,255,0), 1)  # This function draws the rectangle
                                                          # If we want to fill the rectangle then we have
                                                          # replace thickness with cv2.FILLED to fill
                                                          # the rectangle with the same color as the border
cv2.rectangle(img, (300,300), (400,400), (0,255,0), cv2.FILLED)

# Drawing circles
cv2.circle(img, (200,200), 10, (255,0,255), 2)        # This takes parameters image, center point
                                                      # radius in pixels, color, thickness and if you want
                                                      # to fill the circle replace thickness with
                                                      # cv2.FILLED

# Writing text on images
cv2.putText(img, "This text", (300, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)

# This takes in arguments the image, The text to be displayed, the start point of the text,
# the Font type, font scale(can also be in decimal), color, thickness

cv2.imshow("Img",img)
cv2.waitKey(0)  """

####################################### Warp Perspective #######################################
"""
img_ILC = cv2.imread(img_path)

# To extract some non-linear image we need its four coordinates
pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])   # Coordinates of the four points
width, height = 250,350
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]]) # This shows that pts1 is
                                                               # an array with first point as top left
                                                               # last one as bottom right

matrix = cv2.getPerspectiveTransform(pts1, pts2)  # It gets the image inside those coordinates
imgOut = cv2.warpPerspective(img_ILC, matrix, (width,height))

cv2.imshow("Image", img_ILC)
cv2.imshow("Warped", imgOut)
cv2.waitKey(0)  """

############################################ Joining Images ############################################
"""
img_ILC = cv2.imread(img_path)

hor = np.hstack((img_ILC,img_ILC))
vert = np.vstack((img_ILC,img_ILC))

cv2.imshow("Hstacked Image", hor)
cv2.imshow("Vstaked images", vert)
cv2.waitKey(0) """

########################################## Color Detection ##############################################
"""
img_ILC = cv2.imread(img_path)


# Creating a Track Bar, We can use this and do real-time transformation on image
def empty(a):
    pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Bright Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Bright Max", "TrackBars", 255, 255, empty)

while True:
    # HSV images, also known as HSB stands for Hue, Saturation and Brightness
    imgHSV = cv2.cvtColor(img_ILC, cv2.COLOR_BGR2HSV)
    hmin = cv2.getTrackbarPos("Hue Min", "TrackBars")
    hmax = cv2.getTrackbarPos("Hue Max", "TrackBars")
    smin = cv2.getTrackbarPos("Sat Min", "TrackBars")
    smax = cv2.getTrackbarPos("Sat Max", "TrackBars")
    bmin = cv2.getTrackbarPos("Bright Min", "TrackBars")
    bmax = cv2.getTrackbarPos("Bright Max", "TrackBars")
    lower = np.array([hmin, smin, bmin])
    upper = np.array([hmax, smax, bmax])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img_ILC, img_ILC, mask=mask)

    #cv2.imshow("Image", img_ILC)
    cv2.imshow("HSV image", imgHSV)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", imgResult)
    cv2.waitKey(1) """

####################################### Contours/Shape Detection ##########################################
"""
shapes = cv2.imread("C:/Users/amogh/OneDrive/Desktop/shapes.jpg")
shape_contour = shapes.copy()

# The findContours func. takes the image, the mode of retrieval(many other retrieval modes are present)
# RETR_EXTERNAL is good at finding corner points
def getCoutours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 350:
            cv2.drawContours(shape_contour, cnt, -1, (255,0,0), 2) # This func. draws contours on the image
                                                               # It takes the image to draw on, the contour,
                                                               # contour index(if given -1 it will draw
                                                               # contour for all available contours, then
                                                               # it takes the color and thickness
            # Calculates perimeter
            perimeter = cv2.arcLength(cnt, True)

            # Calculates the coordinates of the vertices
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            print(len(approx))
            objCorners = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            objshape = ""
            if objCorners == 3:
                objshape = "Triangle"
            elif objCorners == 4:
                aspRatio = w / float(h)
                if aspRatio > 0.95 and aspRatio < 1.05:
                    objshape = "Square"
                else:
                    objshape = "Rectangle"
            else:
                objshape = "Polygon"

            cv2.rectangle(shape_contour, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(shape_contour, objshape, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)


# To detect the shapes and their corner points we will covert it to gray scale and add a bit of blur
shape_gray = cv2.cvtColor(shapes, cv2.COLOR_BGR2GRAY)
shape_blur = cv2.GaussianBlur(shape_gray, (7,7), 1)

# Detecting edges with canny edge detector
shape_edges = cv2.Canny(shape_blur, 50, 50)

getCoutours(shape_edges)

#cv2.imshow("Shapes", shapes)
cv2.imshow("Gray shapes", shape_gray)
cv2.imshow("Blurred", shape_blur)
cv2.imshow("Edges", shape_edges)
cv2.imshow("Contours", shape_contour)
cv2.waitKey(0)   """

############################################# Face Detection ##############################################

"""
faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
img_ILC = cv2.imread(img_path)
#imgGray = cv2.cvtColor(img_ILC, cv2.COLOR_BGR2GRAY)

# Face Detector

vid = cv2.VideoCapture(0)

while True:
    success, img = vid.read()

    faces = faceCascade.detectMultiScale(img, 1.1, 7)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Face Detector", img)
    #cv2.putText(img, "Amogh", (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    if cv2.waitKey(1) == ord("q"):
        break """

########################### End of Learning, Now go and apply this MF #####################################