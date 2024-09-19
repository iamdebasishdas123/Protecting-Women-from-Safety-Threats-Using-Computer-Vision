**Protecting Women from Safety Threats Using Computer Vision**:

---

# Protecting-Women-from-Safety-Threats-Using-Computer-Vision

## Problem Statement

This project aims to address the issue of women's safety in urban environments by utilizing advanced deep learning and computer vision models to detect potential safety threats in real-time. The solution leverages CCTV footage to detect and analyze safety threats such as violence, the presence of multiple men near a lone woman, suspicious activity, and more, ensuring timely alerts for prevention and intervention.

### SIH Hackathon
This project was initiated as part of the **Smart India Hackathon (SIH)**, where the focus was on developing technology-driven solutions for women’s safety. The proposed solution aims to improve real-time threat detection using AI and alert systems integrated into a CCTV framework.

---

## How to Clone the Repository

To clone the repository to your local machine, use the following command:

```bash
git clone https://github.com/iamdebasishdas123/Protecting-Women-from-Safety-Threats-Using-Computer-Vision.git
```

---

## Documentation

### Architecture

#### **Deep Learning Model Architecture**
![alt text](artifacts\Model.png)

---

### **User Interface**

![image](artifacts\UI.png)

---


The architecture of this project consists of multiple modules for detecting persons, analyzing gender ratios, detecting emotions, and generating SOS alerts when necessary. The system leverages:

- **YOLOv8** for real-time person detection.
- **MediaPipe** for pose and facial expression detection.
- **Transformers** for gender detection.
- **MobileNet** for violence detection.
- **DBSCAN and GRU** for hotspot analysis.

### Technical Approach

1. **Person Detection**: Utilizes YOLOv8 for detecting individuals in CCTV footage and tracking them across frames.
2. **Gender Detection**: Based on Transformer models, it identifies gender to detect if a woman is surrounded by multiple men.
3. **Emotion and Pose Detection**: Uses MediaPipe for facial emotion detection (like fear or distress) and body pose classification (running, walking, standing).
4. **Violence Detection**: MobileNet-based model for identifying violent activity such as fights or physical altercations.
5. **Hotspot Detection**: Using DBSCAN and GRU, the model detects locations with repeated instances of potential danger, identifying unsafe zones or "hotspots."

---

## Links and References

- **Object Detection**: [YOLO Object Detection with OpenCV](https://github.com/yash42828/YOLO-object-detection-with-OpenCV)
- **Gender Classification**: [Hugging Face - Gender Classification](https://huggingface.co/rizvandwiki/gender-classification)
- **Violence Detection**: [Real-Time Violence Detection](https://github.com/abduulrahmankhalid/Real-Time-Violence-Detection)
- **MediaPipe for Pose and Emotion**: [MediaPipe](https://github.com/google-ai-edge/mediapipe)
- **Hotspot Detection**: [Crime Analysis for Hotspot Detection](https://github.com/mcoric96/Crime-analysis)
- **Crime in India Dataset**: [Kaggle Dataset](https://www.kaggle.com/datasets/rajanand/crime-in-india)

---
# Authors:
```bash
Team Name : Guardian Eye
Authors: Debasish Das,Subham , Palmoni, Nidhi, Vandana , Aditya 
Email: iamdebasishdas123@gmail.com
```