import cv2
import imutils

# Initialize capture device
cap = cv2.VideoCapture(1)

# Object's color
objectColorLower = (0, 230, 110)
objectColorUpper = (179, 255, 255)

while True:
    # Read the frame
    _, frame = cap.read()

    # Resize the frame
    frame = imutils.resize(frame, width=600)

    # Gaussian blur // Smoothening
    gaussianBluredImg = cv2.GaussianBlur(frame, (11, 11), 0)

    # BGR to HSV
    hsvImg = cv2.cvtColor(gaussianBluredImg, cv2.COLOR_BGR2HSV)

    # Mask the object's area
    maskedObject = cv2.inRange(hsvImg, objectColorLower, objectColorUpper)

    # Further enhance the masked object
    maskedObject = cv2.erode(maskedObject, None, iterations=2)
    maskedObject = cv2.dilate(maskedObject, None, iterations=2)

    # Find contours
    contours = cv2.findContours(
        maskedObject, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours[-2]

    # Find center of the object
    center = None

    if len(contours) > 0:
        # Object's area
        c = max(
            contours, key=cv2.contourArea
        )  # Gives you the largest object of that color

        ((x, y), radius) = cv2.minEnclosingCircle(c)
        x, y, radius = int(x), int(y), int(radius)

        if radius > 10:  # Make sure the object's big enough
            # Get moments
            moments = cv2.moments(2)

            # Get center point, (0, 0)
            center = (
                int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"]),
            )

            # Drawing circle
            cv2.circle(frame, (x, y), radius, (0, 255, 255), 2)

            # Drawing center point
            cv2.circle(frame, center, 4, (255, 255, 255), -2)

        # print(f"Center - {center}\nx, y - {x}, {y}")

    # Display
    cv2.imshow("Video Stream", frame)

    # Exit condition
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release capture device
cap.release()

# Destroy all windows created
cv2.destroyAllWindows()
