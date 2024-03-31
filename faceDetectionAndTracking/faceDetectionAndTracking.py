import cv2

# Load Haar Cascade algorithm
alg = "haarcascade_frontalface_default.xml"
haarcascade = cv2.CascadeClassifier(alg)

# Initialize cam
cap = cv2.VideoCapture(1)

while True:
    # Read the frame
    _, frame = cap.read()

    # Convert BGR to grayscale
    grayscaleImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get the coordinates of the face
    faceCoordinates = haarcascade.detectMultiScale(grayscaleImg, 1.3, 4)

    # Draw bounding box around the face
    for x, y, w, h in faceCoordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("VideoCapture", frame)

    # Exit condition
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release cam
cap.release()
# Destroy all windows
cv2.destroyAllWindows()
