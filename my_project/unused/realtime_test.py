import tensorflow as tf
import mediapipe as mp
import numpy as np
import cv2
from data_processing import Data_Loader
from preproccing_mediapipe_kinctv2 import mediapipe_to_kinect_v2, Keypoint
from stgc_lstm import GCNLayer


data_loader = Data_Loader('my_project/data/KIMORE/Kimore_ex5')

custom_objects = {'GCNLayer': GCNLayer}
model = tf.keras.models.load_model("my_project/models/my_model_trained_exercise.keras", custom_objects=custom_objects)


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        # Extract pose landmarks
        landmarks = [[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]
        # Convert to Kinect v2 format
        kinect_v2_landmarks = mediapipe_to_kinect_v2(landmarks)
        # Scale landmarks
        scaled_landmarks = data_loader.sc1.transform(np.array(kinect_v2_landmarks).reshape(1, -1)).reshape(-1, 3)
        return scaled_landmarks
    else:
        return None

# Function to predict using the model
def predict_pose(landmarks):
    if landmarks is not None:
        # Reshape and predict
        landmarks = landmarks.reshape(1, -1, 3)
        prediction = model.predict(landmarks)
        return prediction[0][0]
    else:
        return None

# OpenCV video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    landmarks = preprocess_frame(frame)

    # Predict
    if landmarks is not None:
        prediction = predict_pose(landmarks)
        if prediction is not None:
            cv2.putText(frame, f"Prediction: {prediction:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()