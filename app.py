import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

# Initialize MediaPipe models
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Streamlit page configuration
st.title("MediaPipe Models with Streamlit")
st.header("Hand Landmarker Demo with Webcam")

# Function to process a frame and detect hands
def process_frame(frame, model):
    # Convert the color space from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and draw hand landmarks
    results = model.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Convert back to BGR for displaying
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Streamlit component to capture webcam input
st_frame = st.empty()
run = st.checkbox('Run Webcam')
FRAME_WINDOW = st.image([])

# Webcam input
cap = cv2.VideoCapture(0)
while run:
    ret, frame = cap.read()
    if not ret:
        continue

    # Process the frame
    frame = process_frame(frame, hands)
    
    # Display the frame
    FRAME_WINDOW.image(frame, channels="BGR", use_column_width=True)
else:
    st.write('Stopped')

# Clean up
cap.release()
hands.close()
