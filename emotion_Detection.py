import cv2
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model
from keras_preprocessing.image import img_to_array

class EmotionDetector:
    def __init__(self):
        # Load the emotion detection model
        self.emotion_model = load_model("emotion_detection_model_50epochs.h5")
        
        # Labels for the emotion model
        self.emotion_labels = ['Fear', 'Happy', 'Neutral', 'Sad']    

    def detect_emotions(self, face_img):
        # Convert face_img to grayscale
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, (48, 48))
        roi = face_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # Predict emotion
        emotion_preds = self.emotion_model.predict(roi)
        emotion_index = np.argmax(emotion_preds)
        if emotion_index < len(self.emotion_labels):
            emotion_label = self.emotion_labels[emotion_index]
        else:
            emotion_label = "Neutral"
        
        
        return emotion_label

    def draw_results(self, frame, results):
        for result in results:
            x, y, w, h = result['box']
            emotion = result['emotion']
            
            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display emotion label above the face rectangle
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
