"""
Goal: Face recognition
"""

import cv2
import numpy
import os

# Load Haar Cascade frontal face algo.
haarCascade = "haarcascade_frontalface_default.xml"
haarCascadeAlgorithm = cv2.CascadeClassifier(haarCascade)

dataset = "Dataset"  # Directory of the data

# Preparing data for training

(images, labels, names, Id) = ([], [], {}, 0)
# 'images' - all the images at total
# 'labels' - the folder's index
# 'names' - index: name of the person (folder's name)

for (subDirectories, directory, files) in os.walk(dataset):
    # 'subDirectories' - Folders inside the 'dataset'(name of the subject)
    # 'files' - All the images from all the folders('subDirectory')

    for subDirectory in directory:
        # label: name
        names[Id] = subDirectory
        subjectPath = os.path.join(dataset, subDirectory)

        # Into the folder
        for filename in os.listdir(subjectPath):
            path = subjectPath + '/' + filename
            label = Id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        Id += 1

# Load them as array
(images, labels) = [numpy.array(lis) for lis in [images, labels]]
print(images, labels)

(width, height) = (130, 100)

# Load the recognizer
# model = cv2.face.FisherFaceRecognizer_create()

model = cv2.face.LBPHFaceRecognizer_create()

# Training the model
model.train(images, labels)

# Initialize video capture device
cap = cv2.VideoCapture(1)

count = 0

while True:
    _, frame = cap.read()

    # To grayscale
    grayScaleImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection using Haar Cascade
    faceCoordinates = haarCascadeAlgorithm.detectMultiScale(grayScaleImg, 1.3, 5)

    for x, y, w, h in faceCoordinates:
        # Get just the face(crop)
        justFace = grayScaleImg[x: (x + w), y: (y + h)]

        # Draw a rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Resize
        faceResized = cv2.resize(justFace, (width, height))

        # Prediction
        prediction = model.predict(faceResized)
        # print(prediction)

        if prediction[1] < 800:
            cv2.putText(frame, "%s - %0.f" % (names[prediction[0]], prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            print(f"Subject - {names[prediction[0]]}")
            count = 0
        else:
            count += 1
            cv2.putText(frame, "Unknown", (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (64, 64, 64))

            # If a person who is not in the dataset keeps being on cam, save the picture
            if count > 100:
                print("Unknown")
                cv2.imwrite("Unknown.jpg", frame)
                count = 0

    # Display
    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
