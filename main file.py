# gender age and emotio detection by Maryam Siddiqui FJWU
#projct submitted to Sir adeel khalid
#project members  Aimen-jillani,wishma-noor,maryam-siddiqui,Noor-fatima,Kainat-bibi,Natsha-bibi
import cv2
from deepface import DeepFace
import os

# Load Haar cascade classifiers for face and smile detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

# Load pre-trained models for age and gender detection
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load pre-trained models using OpenCV dnn module
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Start capturing video from the default webcam
video = cv2.VideoCapture(0)

padding = 20
max_images = 10
output_folder = 'pictures_of_projects'
os.makedirs(output_folder, exist_ok=True)

# Initialize a counter for naming images cyclically
image_counter = 0

while True:
    ret, frame = video.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = rgb_frame[y:y + h, x:x + w]
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Gender detection
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Age detection
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # Emotion detection
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion'] if result[0]['dominant_emotion'] != 'neutral' else 'Neutral'

        # Smile detection
        smiles = smile_cascade.detectMultiScale(gray_frame, 1.8, 20)
        for sx, sy, sw, sh in smiles:
            image_filename = f'smile_{image_counter % max_images}.jpg'
            path = os.path.join(output_folder, image_filename)
            cv2.imwrite(path, frame)
            print(f"Image saved: {path}")
            image_counter += 1

        # Display the detected information on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, f'Gender: {gender}, Age: {age[1:-1]} years, Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Display the resulting frame with increased window size
    cv2.namedWindow('Real-time Age, Gender, and Emotion Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Real-time Age, Gender, and Emotion Detection', 800, 600)
    cv2.imshow('Real-time Age, Gender, and Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
video.release()
cv2.destroyAllWindows()
