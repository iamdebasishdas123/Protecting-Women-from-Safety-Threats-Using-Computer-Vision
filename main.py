import cv2
import time
import mediapipe as mp
from mtcnn import MTCNN
from Person_Detection import detect_person
from gender_Detection import classify_gender
from emotion_Detection import EmotionDetector
from Centroid_Tracker import CentroidTracker
from SOS_Condition import is_female_surrounded
from pose import detect_action
from Telebot_alert import send_telegram_alert
from facial_expression import classify_face, draw_selected_landmarks

# Initialize video capture and tracker
path = r"C:\Users\Debasish Das\Downloads\WhatsApp Video 2024-08-24 at 13.04.40_79c5f3d4.mp4"
webcam = cv2.VideoCapture(0)
tracker = CentroidTracker()
detector = MTCNN()  # Initialize MTCNN for face detection
emotion_detector = EmotionDetector()  # Initialize the EmotionDetector

mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)
if not webcam.isOpened():
    print("Could not open video")
    exit()

try:
    skip_frame = 7
    frame_count = -1

    while True:
        status, frame = webcam.read()
        if not status:
            print("Failed to read frame from video")
            break

        frame_count += 1
        if frame_count % skip_frame != 0:
            continue

        # Detect persons in the frame
        person_boxes = detect_person(frame)
        n = len(person_boxes)  # stores the number of persons

        # Reset gender counts for the current frame
        male_count = 0
        female_count = 0
        mbbox = []

        # Update tracker with detected person bounding boxes
        objects = tracker.update(person_boxes)
        print(f"Number of detected persons: {len(person_boxes)}")
        print(f"Number of tracked objects: {len(objects)}")

        for i, (objectID, centroid) in enumerate(objects.items()):
            if objectID < len(person_boxes):
                x1, y1, x2, y2 = map(int, person_boxes[i])  # Ensure bounding box values are integers
                person_img = frame[y1:y2, x1:x2]
                
                # Detect faces within the person bounding box using MTCNN
                faces = detector.detect_faces(person_img)

                if faces:
                    face = faces[0]  # Take the first detected face
                    x, y, width, height = face['box']
                    face_img = person_img[y:y+height, x:x+width]

                    # Detect and classify facial expression
                    results = mp_holistic.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    
                    if results.face_landmarks:
                        face_class = classify_face(results.face_landmarks)  # Use the imported classify_face function

                        # Perform gender classification
                        gender_label = classify_gender(face_img)
                        print(f"Gender for objectID {objectID}: {gender_label}")

                        # Perform emotion detection
                        emotion_label = emotion_detector.detect_emotions(face_img)
                        print(f"Emotion for objectID {objectID}: {emotion_label}")
                        

                        if gender_label:
                            # Update counts based on gender
                            if 'male' in gender_label:
                                male_count += 1
                                mbbox.append(person_img)
                            elif 'female' in gender_label:
                                female_bbox = person_img
                                female_count += 1

                            # Detect and classify pose
                            pose_action = detect_action(results.pose_landmarks)

                            # Combine all detected features into the label
                            label = f'ID {objectID}: {gender_label}, {face_class}, {emotion_label}, {pose_action}'

                            # Draw facial landmarks
                            draw_selected_landmarks(face_img, results.face_landmarks)

                            # Alert condition: Female detected alone at night
                            if n == 1 and 'female' in gender_label and (time.localtime().tm_hour >= 18 or time.localtime().tm_hour < 6):
                                send_telegram_alert(frame, "Female detected alone at night!")
                                print("Alert sent: Female detected alone at night.")
                        else:
                            label = f'ID {objectID}: Unknown'
                    else:
                        label = f'ID {objectID}: Person'

                    # Annotate the frame
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, tuple(map(int, centroid)), 4, (255, 0, 0), -1)
                else:
                    print(f"Warning: No face detected for object ID {objectID}")
                    
        # This part detects and alerts if a female is surrounded by males.
        if female_count == 1 and n > 2 and (face_class == 'Fear' or face_class == 'Distress'):
            if is_female_surrounded(female_bbox, mbbox):
                send_telegram_alert(frame, "Female surrounded by men, potential danger detected!")
                print("Alert sent: Female surrounded by men.")
        
        # Display the counts of males, females, and persons for the current frame
        count_text = f'Males: {male_count}  Females: {female_count}  Total Persons: {n}'
        cv2.putText(frame, count_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Display the annotated frame
        cv2.imshow("Webcam/Video Feed", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release resources
    webcam.release()
    cv2.destroyAllWindows()
