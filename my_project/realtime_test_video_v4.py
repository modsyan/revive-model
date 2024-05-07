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

def preprocess_frame(frame_list):
    frame_list_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frame_list]

    # Process frames
    frame_list_results = [pose.process(frame_rgb) for frame_rgb in frame_list_rgb]

    # Extract landmarks and filter out frames where no landmarks are detected
    frame_list_landmarks = []
    for frame_results in frame_list_results:
        if frame_results.pose_landmarks:
            landmarks = [[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in frame_results.pose_landmarks.landmark]
            frame_list_landmarks.append(landmarks)

    # Convert landmarks to Kinect v2
    frame_list_vkv2 = []
    for landmarks in frame_list_landmarks:
        vkv2 = mediapipe_to_kinect_v2(landmarks)
        frame_list_vkv2.append(vkv2)

    # Convert to numpy array
    frame_list_vkv2_array = [np.array(vkv2) for vkv2 in frame_list_vkv2]

    # Extract only 3D coordinates
    frame_list_vkv2_3d = [vkv2_array[:, :3] for vkv2_array in frame_list_vkv2_array]

    # Reshape to match the input shape of the model
    frame_list_result = [np.array([coord for point in vkv2_3d for coord in point]) for vkv2_3d in frame_list_vkv2_3d]

    # Convert to numpy array
    frame_list_result_array = np.array(frame_list_result)
    print("frame_list_result_array.shape: ", frame_list_result_array.shape)

    test_data_loader = Test_Data_Loader(frame_list_result_array)

    return test_data_loader

def predict_pose(landmarks):
    if landmarks is not None:
        # prediction = model.predict(landmarks)[0][0]
        # print("Prediction: ", prediction)
        # return prediction
        return None
    else:
        return None

# cap = cv2.VideoCapture('my_project/data/VID20240328063258.mp4')
cap = cv2.VideoCapture('my_project/data/test_st.mp4')
# cap = cv2.VideoCapture(0)

frames_collector = []
predictions = []
score = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if (len(frames_collector) < 100): 
        continue

    frames_collector.append(frame)
    batch_size = 100
    landmarks = preprocess_frame(frames_collector)

    for i in range(landmarks.scaled_x.shape[0]):
        prediction = model.predict(landmarks.scaled_x[i].reshape(1,landmarks.scaled_x[i].shape[0],landmarks.scaled_x[i].shape[1],landmarks.scaled_x[i].shape[2]))
        predictions.append(prediction[0,0])
    score = round(prediction[0][0] * 100) if prediction[0][0] > 0 else 0
    print("Prediction: ", score)

    cv2.putText(frame, f"Prediction: {score}%", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Exercise 5", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow('output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()