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
model = tf.keras.models.load_model("my_project/models/my_model_trained_exercise.keras", custom_objects=custom_objects)


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        landmarks = [[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]
        kinect_v2_landmarks = mediapipe_to_kinect_v2(landmarks)
        # print(kinect_v2_landmarks)

        kinect_v2_landmarks = np.array(kinect_v2_landmarks)
        # print("Shape of kinect_v2_landmarks: ", np.array(kinect_v2_landmarks).shape)

        # selected_landmarks = np.array(selected_landmarks).reshape(1, -1)
        # scaled_landmarks = data_loader.sc1.transform(selected_landmarks)
        # scaled_landmarks = scaled_landmarks.reshape(-1, 3)
        # return data.scaled_x[i].reshape(1,data.scaled_x[i].shape[0],data.scaled_x[i].shape[1],data.scaled_x[i].shape[2])

        # data_as_video = Test_Data_Loader(kinect_v2_landmarks)
        # frameProcess = FramePreprocessing(kinect_v2_landmarks).scaled_x
        # test_processing = TestProcessing(kinect_v2_landmarks).scaled_x

        # return test_processing

        # return kinect_v2_landmarks[:, :-1]

        # Extract the first three columns (assuming x, y, z coordinates)
        kinect_v2_landmarks_3d = kinect_v2_landmarks[:, :3]

        # Reshape the array to have an extra dimension at the beginning
        kinect_v2_landmarks_reshaped = kinect_v2_landmarks_3d[np.newaxis, np.newaxis, :, :]



        return kinect_v2_landmarks_reshaped

    else:
        return None

def mock_frame():
    num_frames = 10
    num_joints = 25 
    num_channels = 3 
    time_steps = 1
    batch_size = 1

    mock_frame = np.random.rand(batch_size, time_steps,num_frames, num_joints * num_channels)
    print (mock_frame)

    for i in range(batch_size):
        for j in range(time_steps):
            for k in range(num_frames):
                mock_frame[i,j,k] = None
                for l in range(num_joints * num_channels):
                    mock_frame[i,j,k,l] = None
    return mock_frame


def predict_pose(landmarks):
    if landmarks is not None:
        prediction = model.predict(landmarks)[0][0]
        print("Prediction: ", prediction)
        return prediction
    else:
        return None

predictions = []
cap = cv2.VideoCapture('my_project/data/squat_test.mp4')

while True:

    # ret, frame = cap.read()
    # if not ret:
    #     break

    array_of_100_frames = []
    while len(array_of_100_frames) < 100:
        ret, frame = cap.read()
        if not ret:
            break
        array_of_100_frames.append(frame)


    landmarks = preprocess_frame(frame)
    preprocess_frames_of_100 = []
    
    prediction = predict_pose(landmarks) if landmarks is not None else None
    score = prediction if prediction is not None else 0

    predictions.append(score)

    cv2.putText(frame, f"Prediction: {score:f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Exercise 5", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow('output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()