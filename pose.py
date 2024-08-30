import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def detect_action(pose_landmarks):
    if not pose_landmarks:
        return "Unknown"

    # Points used for simple action detection
    left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

    # Calculate distances between ankles and knees
    ankle_distance = calculate_distance(left_ankle, right_ankle)
    knee_distance = calculate_distance(left_knee, right_knee)

    # Heuristic to determine if the person is walking
    if ankle_distance > 0.5 and knee_distance > 0.5:
        return "Walking"
    else:
        return "Standing"