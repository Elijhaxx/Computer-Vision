import facial_emotion_recognition
import cv2

emotionRecognition = facial_emotion_recognition.EmotionRecognition(device="gpu")
cap = cv2.VideoCapture(1)

while True:
    _, frame = cap.read()
    frame = emotionRecognition.recognise_emotion(frame, return_type="BGR")

    cv2.imshow("VideoStream", frame)
    if cv2.waitKey(1) == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
