import cv2
import imutils

# Load the cam
cap = cv2.VideoCapture(0)

firstFrame = None

area = 500

while True:
    # Read the frame
    _, frame = cap.read()

    text = "Normal"

    # Image resizing
    resizedFrame = imutils.resize(frame, width=500)

    # BGR to greyscale image
    greyscaleImg = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)

    # Applying Gaussian blur filter
    gaussianImg = cv2.GaussianBlur(greyscaleImg, (21, 21), cv2.BORDER_DEFAULT)

    # Assign firstImage
    if firstFrame is None:
        firstFrame = gaussianImg
        continue

    # Absolute difference b/w first and the current frame
    imgDifference = cv2.absdiff(firstFrame, gaussianImg)

    # Convert to threshold image
    threshImg = cv2.threshold(imgDifference, 25, 255, cv2.THRESH_BINARY)[1]

    # Further enhance the threshold image w dilate
    dilatedImg = cv2.dilate(threshImg, None, iterations=2)

    # Finding contours
    contours = cv2.findContours(dilatedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Grab contours
    contours = imutils.grab_contours(contours)

    # Loop through each contour
    for contour in contours:
        if cv2.contourArea(contour) < area:
            continue
        # Coordinates of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update text
        text = "Moving Object Detected"

    print(text)
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("VideoStream", frame)

    # Termination condition - 'q'
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release cam
cap.release()

# Destroy all windows
cv2.destroyAllWindows()