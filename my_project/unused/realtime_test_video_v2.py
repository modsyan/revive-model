import tensorflow as tf
import mediapipe as mp
import numpy as np
import cv2
from data_processing import Data_Loader, Test_Data_Loader
from frame_preprocessing import FramePreprocessing
from test_dataset_processing import TestProcessing
from preproccing_mediapipe_kinctv2 import mediapipe_to_kinect_v2, Keypoint
from stgc_lstm import GCNLayer

data_loader = Data_Loader('my_project/data/KIMORE/Kimore_ex5')

custom_objects = {'GCNLayer': GCNLayer}
model = tf.keras.models.load_model("my_project/models/my_model_trained_exercisetest500.keras", custom_objects=custom_objects)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        landmarks = [[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]
        kinect_v2_landmarks = mediapipe_to_kinect_v2(landmarks)
        kinect_v2_landmarks = np.array(kinect_v2_landmarks)
        kinect_v2_landmarks_3d = kinect_v2_landmarks[:, :3]
        kinect_v2_landmarks_reshaped = kinect_v2_landmarks_3d[np.newaxis, np.newaxis, :, :]
        # return kinect_v2_landmarks_reshaped

        # zero_frame = np.zeros((1, kinect_v2_landmarks_3d.shape[1]*kinect_v2_landmarks_3d.shape[0]))
        # for body_part in range(kinect_v2_landmarks_3d.shape[0]):
        #     for channel in range(kinect_v2_landmarks_3d.shape[1]):
        #         zero_frame[0, body_part + channel] = kinect_v2_landmarks_3d[body_part, channel]
        # print(zero_frame)

        result_list = [[coord for point in kinect_v2_landmarks_3d for coord in point]]
        result_array = np.array(result_list)


        print("From here")
        print("result_array.shape", result_array.shape)
        print("result_array", result_array)
        test_data_loader = Test_Data_Loader(result_array)

        return test_data_loader
    else:
        return None

def predict_pose(landmarks):
    if landmarks is not None:
        prediction = model.predict(landmarks)[0][0]
        print("Prediction: ", prediction)
        return prediction
    else:
        return None

cap = cv2.VideoCapture('my_project/data/VID20240328063258.mp4')
# cap = cv2.VideoCapture(0)

while True:
    # array_of_100_frames = []
    # while len(array_of_100_frames) < 100:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     array_of_100_frames.append(frame)

    # landmarks = preprocess_frame(frame)
    # preprocess_frames_of_100 = []

    ret , frame = cap.read()
    if not ret:
        break

    landmarks = preprocess_frame(frame) 
    # prediction = predict_pose(landmarks) if landmarks is not None else None
    # score = prediction if prediction is not None else 0
    # predictions.append(score)

    predictions = []
    # for i in range(landmarks.scaled_x.shape[0]):
    #     prediction = model.predict(landmarks.scaled_x[i].reshape(1,landmarks.scaled_x[i].shape[0],landmarks.scaled_x[i].shape[1],landmarks.scaled_x[i].shape[2]))
    #     predictions.append(prediction[0,0])

    score = np.mean(predictions)

    cv2.putText(frame, f"Prediction: {score:f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Exercise 5", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow('output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()