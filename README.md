# OpenCV-Projects

OpenCV Projects include:
  - Document Scanner
  - Number Plate Detector
  - Virtual Writing
  - OpenCV Basics file
## Document Scanner 
      Document Scanner uses Warping techniques to extract region where the document is present and crops it
      off and presents only the cropped off document. It can be used in Real-time also, given webcam is of better quality
      so that the document that is being cropped off can be clearly differentiated from the other surfaces.
      The process of getting the cropped of document is as follows:
          1. Preprocessing
              In preprocessing we usually convert the images obtained from the webcam to grayscale images. 
              We convert the images to grayscale as canny edges can be better identified when the image is 
              a grayscale image. We then add a bit of blur to reduce the noise. Then identify the canny edges.
              Also to distinctly identify the edges we dilate and erode each for 1 iterations.
          
          2. Get the Contours
              Getting the contours is the most important part because this is the part where we differentiate the 
              usefull document and useless background. We might be getting many contours so the approach is to get
              the contour with maximum area. Also to reduce the number of contours we get we can have a threshhold 
              value for the area.
              
          3. Reorder the corner points
              The corner points of the biggest contour will be at random, meaning the first corner point need not be
              the top-left corner point as so is the case with other corners also. 
              
          4. Warp Perspective
              Now as we have the ordered corner points of the biggest contour we can get the warped image. To do so we 
              have to first configure the cv2.warpPerspectiveTransorm() with the points. Then we get a matrix of points
              which we pass to cv2.warpPerspective() to get the warped image.
              
## Number Plate Detector
      Number Plate Detector can be easily made using the haarcascade that has already been trained using different kinds 
      of number plates. Haarcascading is a method of object detection. We only have to initialize a cascade classifier and
      we will be good to go.
      
## Virtual Writing
      Virtual Writing is essentially drawing on the screen. We do this by identifying the color of the object with which a 
      person wants to write. Then we can get it's contours and then draw on the screen.
      To do so we have to follow some steps:
          1. Finding the colors
              We will have a set of predefined colors that we know. Color detection is done using the HSV convention, so we
              convert our image to a HSV image. For every color we will have a range of HSV values. Then for each color we will
              check if the color is present. We do this using th cv2.inRange() method which returns a mask. This mask is used to
              know if the color is present or not. 
          
          2. Get the Contours
              We use the mask to find out whether the color is present or not. If present we get it's contours and also the coordinates.
              As we can get multiple contours we can have a threshold on the area of the contours found.
              
          3. Append the coordinates
              We append the coordinates if coordinates aren't zeros(means the color wasn't found). Also while appending the coordinates we 
              append the index of it's color so that we can use it to draw the same color on the screen.
              
          4. Drawing
              We will be having a global list of coordinates with colors that are found. We will be appending all the values to this list so
              whenever a new color is discovered we can draw it. We also make sure that we do identify colors by checking the length of the 
              global list, if it's length turns out to be zero it would mean that we haven't detected any color. Then for every coordinate 
              we draw a circle on the screen.
              
  ## OpenCV Basics
      This file contains the basic overview of most used methods that are used in image manipulation. It aslo includes some information in 
      comment format. It also gives an introduction identifying shapes, getting cotours, and warping images.
