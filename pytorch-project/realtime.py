import cv2
import mediapipe as mp
import numpy as np
import torch

# Load the exercise recognition model
model = torch.load('./models/all_exercises_model.pt')
model.eval()

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to classify exercise
def classify_exercise(pose_keypoints):
    # Preprocess pose keypoints (reshape, normalize, etc. to match model input)
    # Run the model inference
    with torch.no_grad():
        outputs = model(pose_keypoints)
        _, predicted_label = torch.max(outputs, 1)
    return encoder.inverse_transform(predicted_label.numpy())[0]  # Convert label to exercise name

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB and process
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect poses in the frame
    results = pose_model.process(image_rgb)

    # If poses are detected
    if results.pose_landmarks:
        # Extract pose keypoints
        pose_keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]).astype(np.float32)
        
        # Classify the exercise
        exercise = classify_exercise(pose_keypoints)
        
        # Draw the exercise label on the frame
        cv2.putText(frame, exercise, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Exercise Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()