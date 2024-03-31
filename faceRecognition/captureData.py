"""
Goal: Capture & save 50 frontal face images of the subject in their respective folder.
"""

import cv2
import os

# Data save location
dataset = "Dataset"
person = input("Subject's name - ")

path = os.path.join(dataset, person)

if not os.path.isdir(path):
    os.mkdir(path)

# Load Haar Cascade frontal face algo.
haarCascade = "haarcascade_frontalface_default.xml"
haarCascadeAlgorithm = cv2.CascadeClassifier(haarCascade)

# Initialize video capture device
cap = cv2.VideoCapture(1)

(imgWidth, imgHeight) = (130, 100)

count = 1

while count < 31:
    _, img = cap.read()
    print(f"Count - {count}")

    # To grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get face's position
    faceCoordinates = haarCascadeAlgorithm.detectMultiScale(grayImg, 1.3, 4)

    for x, y, w, h in faceCoordinates:
        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Get just the face
        justFace = grayImg[y: (y + h), x: (x + w)]

        # Resize
        justFaceResized = cv2.resize(justFace, (imgWidth, imgHeight))

        # Save
        cv2.imwrite("%s/%s.png" % (path, count), justFaceResized)
        count += 1

    cv2.imshow("VideoStream", img)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()