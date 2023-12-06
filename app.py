import streamlit as st
import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe models
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection

hands = mp_hands.Hands()
face_detection = mp_face_detection.FaceDetection()

# Streamlit page configuration
st.title("MediaPipe Models with Streamlit")
st.sidebar.title("Model Selection")
model_choice = st.sidebar.radio("Choose a Model", ("Hand Landmarker", "Face Detector"))

# Function to process a frame with Hand Landmarker
def process_hand_frame(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Function to process a frame with Face Detector
def process_face_frame(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.process(frame)
    if results.detections:
        for detection in results.detections:
            mp.solutions.drawing_utils.draw_detection(frame, detection)
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

    # Display the frame
    FRAME_WINDOW.image(frame, channels="BGR", use_column_width=True)
else:
    st.write('Stopped')

# Clean up
cap.release()
hands.close()
face_detection.close()
