import cv2
from transformers import pipeline
from PIL import Image

# Initialize the gender classification pipeline
gender_classifier = pipeline("image-classification", model="rizvandwiki/gender-classification")

def classify_gender(face_image):
    # Check if the face image is too small
    if face_image.shape[0] < 10 or face_image.shape[1] < 10:
        return None, None

    # Convert the face image from BGR to RGB
    rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)

    # Perform gender classification
    results = gender_classifier(images=pil_image)

    # Extract the predicted label and confidence score
    label = results[0]['label']
    confidence = results[0]['score']

    return label, round(confidence,2)

