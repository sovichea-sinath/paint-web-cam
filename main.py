
import numpy as np
import cv2
from collections import deque

"""
The blueLower and the blueUpper numpy arrays help us in finding the blue colored cap.
The kernal helps in smoothing blue cap once found.
The bpoints, gpoints, rpoints and ypoints deques are used to store the points drawn
on the screen of color blue, green, red and yellow respectively.
"""
# Define the upper and lower boundaries for a color to be considered "Blue"
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

# Initialize deques to store different colors in different arrays
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

# Initialize an index variable for each of the colors 
bindex = 0
gindex = 0
rindex = 0
yindex = 0

# Just a handy array and an index variable to get the color-of-interest on the go
# Blue, Green, Red, Yellow respectively
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)] 
colorIndex = 0

"""
This is a painful thing to do. We have to manually set the coordinates of each of
the color boxes on the frame. We use the OpenCV function cv2.rectangle()
to draw the boxes.
"""


# Create a blank white image
paintWindow = np.zeros((471,636,3)) + 255

# Draw buttons like colored rectangles on the white image
paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), colors[1], -1)
paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), colors[3], -1)

# Label the rectanglular boxes drawn on the image
cv2.putText(paintWindow, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)

# Create a window to display the above image (later)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

"""
Now we use the OpenCV function cv2.VideoCapture() method to read a video,
frame by frame (using a while loop), either from a video file or from
a webcam in real-time. In this case, we pass 0 to the method to read from a webcam.
"""
# Load the video
camera = cv2.VideoCapture(0)

# Keep looping
while True:
  # Grab the current paintWindow
  (grabbed, frame) = camera.read()
  frame = cv2.flip(frame, 1)
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # wait for ESC key
  key = cv2.waitKey(20)

  # Check to see if we have reached the end of the video (useful when input is a video file not a live video stream)
  if not grabbed or key == 27:
    break    

  # Add the same paint interface to the camera feed captured through the webcam (for ease of usage)
  frame = cv2.rectangle(frame, (40,1), (140,65), (122,122,122), -1)
  frame = cv2.rectangle(frame, (160,1), (255,65), colors[0], -1)
  frame = cv2.rectangle(frame, (275,1), (370,65), colors[1], -1)
  frame = cv2.rectangle(frame, (390,1), (485,65), colors[2], -1)
  frame = cv2.rectangle(frame, (505,1), (600,65), colors[3], -1)
  cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
  cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
  cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
  cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
  cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)

  """
  Once we start reading the webcam feed, we constantly look for a blue color
  object in the frames with the help of cv2.inRange() method and use the blueUpper
  and blueLower variables initialized in Step 0. Once we find the contour,
  we do a series of image operations and make it smooth.
  They just makes our lives easier. If you want to know
  more about these operations — erode, morph and dilate, check this out.
  """

  # Determine which pixels fall within the blue boundaries and then blur the binary image
  blueMask = cv2.inRange(hsv, blueLower, blueUpper)
  blueMask = cv2.erode(blueMask, kernel, iterations=2)
  blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
  blueMask = cv2.dilate(blueMask, kernel, iterations=1)

  # Find contours in the image
  (cnts, hierarchy) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  """
  Once we find the contour (the if condition passes when a contour is found),
  we use the center of the contour (blue cap) to draw on the screen as it moves.
  The following code does the same.
  """

  # Check to see if any contours (blue stuff) were found
  if len(cnts) > 0:
    # Sort the contours and find the largest one -- we assume this contour correspondes to the area of the bottle cap
    cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    # Get the radius of the enclosing circle around the found contour
    ((x, y), radius) = cv2.minEnclosingCircle(cnt)
    # Draw the circle around the contour
    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
    # Get the moments to calculate the center of the contour (in this case a circle)
    M = cv2.moments(cnt)
    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

    """
    The above code finds the contour (the largest one),
    draws a circle around it using the cv2.minEnclosingCircle() and cv2.circle()
    methods, gets the center of the contour found with the help of cv2.moments() method.
    """

    """
    Now we start tracking coordinates of each and every point the center of the contour
    touches on the screen, along with its color. We store these set of points of
    different colors in different deques (bpoints, gpoints etc.).
    When the center of the contour touches one of the colored boxes we put on the
    screen in Step 1, we store the points in its respective color deque.
    """

    if center[1] <= 65:
      if 40 <= center[0] <= 140: # Clear All
          bpoints = [deque(maxlen=512)]
          gpoints = [deque(maxlen=512)]
          rpoints = [deque(maxlen=512)]
          ypoints = [deque(maxlen=512)]

          bindex = 0
          gindex = 0
          rindex = 0
          yindex = 0

          paintWindow[67:,:,:] = 255
      elif 160 <= center[0] <= 255:
              colorIndex = 0 # Blue
      elif 275 <= center[0] <= 370:
              colorIndex = 1 # Green
      elif 390 <= center[0] <= 485:
              colorIndex = 2 # Red
      elif 505 <= center[0] <= 600:
              colorIndex = 3 # Yellow
    else :
      if colorIndex == 0:
          bpoints[bindex].appendleft(center)
      elif colorIndex == 1:
          gpoints[gindex].appendleft(center)
      elif colorIndex == 2:
          rpoints[rindex].appendleft(center)
      elif colorIndex == 3:
          ypoints[yindex].appendleft(center)
      # Append the next deque when no contours are detected (i.e., bottle cap reversed)
      else:
        bpoints.append(deque(maxlen=512))
        bindex += 1
        gpoints.append(deque(maxlen=512))
        gindex += 1
        rpoints.append(deque(maxlen=512))
        rindex += 1
        ypoints.append(deque(maxlen=512))
        yindex += 1

  """
  So far we stored all the points in their respective color deques.
  Now we just join them using a line of their own color.
  The OpenCV function cv2.line() comes in handy for us to do that.
  The following code does the same.
  """

  # Draw lines of all the colors (Blue, Green, Red and Yellow)
  points = [bpoints, gpoints, rpoints, ypoints]
  for i in range(len(points)):
      for j in range(len(points[i])):
          for k in range(1, len(points[i][j])):
              if points[i][j][k - 1] is None or points[i][j][k] is None:
                  continue
              cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
              cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

  # Show the frame and the paintWindow image
  cv2.imshow("Tracking", frame)
  cv2.imshow("Paint", paintWindow)

# Cleanup code
camera.release()
cv2.destroyAllWindows()
