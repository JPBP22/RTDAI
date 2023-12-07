import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import cv2
import yaml
import tempfile
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import csv
import datetime
import os

# Load the configuration from the YAML file for authentication
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Create the authenticator object using the loaded configuration
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Initialize MediaPipe models
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# Processing functions for each model
def process_hand_frame(frame, model):
    results = model.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame

def process_face_frame(frame, model):
    results = model.process(frame)
    if results.detections:
        for detection in results.detections:
            mp.solutions.drawing_utils.draw_detection(frame, detection)
    return frame

def process_face_mesh_frame(frame, model):
    results = model.process(frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
    return frame

def process_pose_frame(frame, model):
    results = model.process(frame)
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return frame

def process_frame(frame, model_choice):
    if model_choice == "Hand Landmarker":
        with mp_hands.Hands() as hands:
            return process_hand_frame(frame, hands)
    elif model_choice == "Face Detector":
        with mp_face_detection.FaceDetection() as face_detection:
            return process_face_frame(frame, face_detection)
    elif model_choice == "Face Landmark":
        with mp_face_mesh.FaceMesh() as face_mesh:
            return process_face_mesh_frame(frame, face_mesh)
    elif model_choice == "Pose Landmark":
        with mp_pose.Pose() as pose:
            return process_pose_frame(frame, pose)

# Video Processor Class
class VideoProcessor(VideoTransformerBase):
    def __init__(self, model_choice):
        self.model_choice = model_choice

    def transform(self, frame):
        img = frame.to_ndarray(format="rgb24")
        return process_frame((cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), self.model_choice)

# Function to log the data
def log_data(username, model_choice=None, source=None, timestamp=None, action=None):
    log_file = 'usage_log.csv'
    log_entry = [username, model_choice, source, timestamp, action]

    # Check if the log file exists and write headers if it doesn't
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Username', 'Model Choice', 'Source', 'Timestamp', 'Action'])
        writer.writerow(log_entry)

# Initialize session state for login and logout logging
if 'logged_login' not in st.session_state:
    st.session_state['logged_login'] = False
if 'logged_logout' not in st.session_state:
    st.session_state['logged_logout'] = False

# Authentication
name, authentication_status, username = authenticator.login("Login", "main")

# Log login action
if authentication_status and not st.session_state['logged_logout']:
    log_data(username, action='login', timestamp=datetime.datetime.now())
else: 
    st.session_state['logged_login'] = True

# Main application logic
if authentication_status:
    # Streamlit page configuration
    st.title(f"MediaPipe Models with Streamlit - Welcome {name}")
    st.sidebar.title("Model Selection")
    model_choice = st.sidebar.radio("Choose a Model", ("Hand Landmarker", "Face Detector", "Face Landmark", "Pose Landmark"))

    # Video Source Selection
    source = st.sidebar.radio("Choose Video Source", ("Webcam", "Video File"))

    # Video Processing based on source
    if source == "Webcam":
        webrtc_streamer(key="example", video_processor_factory=lambda: VideoProcessor(model_choice))
        log_data(username, model_choice, "Webcam", datetime.datetime.now())
    elif source == "Video File":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            # Temporary file to store the uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                tmpfile.write(uploaded_file.read())
                video_path = tmpfile.name

            cap = cv2.VideoCapture(video_path)
            frame_placeholder = st.empty()  # Placeholder for video frames

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = process_frame(frame, model_choice)
                frame_placeholder.image(processed_frame, channels="RGB", use_column_width=True)

            cap.release()
        log_data(username, model_choice, "Video File", datetime.datetime.now())

if 'Logout' in st.session_state and st.session_state['logged_login']:
    log_data(username, action='logout', timestamp=datetime.datetime.now())
    if  st.session_state['logged_logout']:
        st.session_state['logged_logout'] = True
    del st.session_state['Logout']

if st.session_state["authentication_status"]:
    st.session_state['Logout'] = authenticator.logout('Logout', 'main', key='unique_key')
    
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
