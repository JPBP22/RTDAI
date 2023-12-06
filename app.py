import streamlit as st
import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe models
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

hands = mp_hands.Hands()
face_detection = mp_face_detection.FaceDetection()
face_mesh = mp_face_mesh.FaceMesh()
pose = mp_pose.Pose()
holistic = mp_holistic.Holistic()

# Streamlit page configuration
st.title("MediaPipe Models with Streamlit")
st.sidebar.title("Model Selection")
model_choice = st.sidebar.radio("Choose a Model", ("Hand Landmarker", "Face Detector", 
                                                   "Face Landmark", "Pose Landmark", "Hand Gesture"))

# Processing functions for each model
def process_hand_frame(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def process_face_frame(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.process(frame)
    if results.detections:
        for detection in results.detections:
            mp.solutions.drawing_utils.draw_detection(frame, detection)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def process_face_mesh_frame(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.process(frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def process_pose_frame(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.process(frame)
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def process_hand_gesture_frame(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.process(frame)
    if results.left_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Streamlit component to capture webcam input
run = st.checkbox('Run Webcam')
FRAME_WINDOW = st.image([])

# Webcam input
cap = cv2.VideoCapture(0)
while run:
    ret, frame = cap.read()
    if not ret:
        continue

    # Process the frame based on model choice
    if model_choice == "Hand Landmarker":
        frame = process_hand_frame(frame, hands)
    elif model_choice == "Face Detector":
        frame = process_face_frame(frame, face_detection)
    elif model_choice == "Face Landmark":
        frame = process_face_mesh_frame(frame, face_mesh)
    elif model_choice == "Pose Landmark":
        frame = process_pose_frame(frame, pose)
    elif model_choice == "Hand Gesture":
        frame = process_hand_gesture_frame(frame, holistic)

    # Display the frame
    FRAME_WINDOW.image(frame, channels="BGR", use_column_width=True)
else:
    st.write('Stopped')

# Clean up
cap.release()
hands.close()
face_detection.close()
face_mesh.close()
pose.close()
holistic.close()
