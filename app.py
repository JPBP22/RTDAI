import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe models
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# Streamlit page configuration
st.title("MediaPipe Models with Streamlit")
st.sidebar.title("Model Selection")
model_choice = st.sidebar.radio("Choose a Model", ("Hand Landmarker", "Face Detector", "Face Landmark", "Pose Landmark"))

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

# Video Processor Class
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if model_choice == "Hand Landmarker":
            with mp_hands.Hands() as hands:
                img = process_hand_frame(img, hands)
        elif model_choice == "Face Detector":
            with mp_face_detection.FaceDetection() as face_detection:
                img = process_face_frame(img, face_detection)
        elif model_choice == "Face Landmark":
            with mp_face_mesh.FaceMesh() as face_mesh:
                img = process_face_mesh_frame(img, face_mesh)
        elif model_choice == "Pose Landmark":
            with mp_pose.Pose() as pose:
                img = process_pose_frame(img, pose)

        return img

# Streamlit component to capture webcam input
webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
