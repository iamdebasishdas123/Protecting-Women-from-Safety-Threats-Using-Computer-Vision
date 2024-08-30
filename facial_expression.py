import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Holistic.
mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define the indices for a medium number of facial landmarks.
selected_indices = [
    33, 133, 362, 263,  # Eye corners (left & right)
    70, 105, 336, 300,  # Eyebrow (inner & outer points)
    1, 4, 197, 195, 5,  # Nose tip and nostrils
    61, 291, 13, 14,  # Mouth corners and lip centers
    10, 152, 234, 454  # Jawline and cheekbones
]

def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points."""
    angle = math.degrees(math.atan2(p3.y - p2.y, p3.x - p2.x) - 
                         math.atan2(p1.y - p2.y, p1.x - p2.x))
    return abs(angle)

def classify_face(landmarks):
    if landmarks:
        # Retrieve relevant landmarks.
        left_eye_inner = landmarks.landmark[133]
        right_eye_inner = landmarks.landmark[362]
        left_eyebrow_inner = landmarks.landmark[70]
        right_eyebrow_inner = landmarks.landmark[300]
        upper_lip = landmarks.landmark[13]
        lower_lip = landmarks.landmark[14]
        left_mouth_corner = landmarks.landmark[61]
        right_mouth_corner = landmarks.landmark[291]

        # Calculate smile angles (left and right).
        smile_angle = calculate_angle(left_mouth_corner, upper_lip, right_mouth_corner)

        # Calculate eyebrow-eye angles (left and right).
        left_eyebrow_eye_angle = calculate_angle(left_eyebrow_inner, left_eye_inner, right_eye_inner)
        right_eyebrow_eye_angle = calculate_angle(right_eyebrow_inner, right_eye_inner, left_eye_inner)

        # Calculate mouth openness (Y distance between upper and lower lips).
        mouth_openness = abs(upper_lip.y - lower_lip.y)
        smile_curve = (left_mouth_corner.y + right_mouth_corner.y) / 2 - (upper_lip.y + lower_lip.y) / 2
        mouth_width = abs(left_mouth_corner.x - right_mouth_corner.x)

        # Define thresholds for each emotion.
        happy_threshold = mouth_width > 0.05 and (smile_curve < 0 and mouth_openness < 0.02)  # Smile curve and mouth width.   # Smile angle is small for a happy face.
        fear_threshold = (left_eyebrow_eye_angle > 20 or right_eyebrow_eye_angle > 20) and mouth_openness > 0.03  # Raised eyebrows and wide-open mouth.
        distress_threshold = (left_eyebrow_eye_angle > 10 or right_eyebrow_eye_angle > 10) and mouth_openness > 0.02  # Slightly raised eyebrows and open mouth.
        neutral_threshold = not (happy_threshold or fear_threshold or distress_threshold)  # If none of the other conditions are met.

        # Classification logic.
        if happy_threshold:
            return "Happy"
        elif fear_threshold:
            return "Fear"
        elif distress_threshold:
            return "Distress"
        elif neutral_threshold:
            return "Neutral Face"
    return "Face Not Detected"

def draw_selected_landmarks(image, landmarks):
    """Draw a medium number of facial landmarks."""
    for index in selected_indices:
        landmark = landmarks.landmark[index]
        h, w, _ = image.shape
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)
