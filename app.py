import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("C:\Users\ADVICE_003\Desktop\New folder\test_model_deploy-1\best.pt")  # Ensure best.pt is in the same directory

# Streamlit UI
st.title("🍛 Thai Food Classifier with YOLO")
st.write("Upload an image of Thai food, and the model will classify it!")

# Confidence threshold slider
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    img = np.array(image)

    # YOLO Prediction with confidence filtering
    results = model(img)

    # Process results
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        scores = result.boxes.conf  # Confidence scores
        classes = result.boxes.cls.int().tolist()  # Class indices

        # Class names
        class_names = ["Pad Thai", "Tom Yum", "Som Tum", "Green Curry", "Massaman Curry",
                       "Khao Soi", "Moo Ping", "Pad Krapow", "Kai Jeow", "Khao Man Gai"]

        # Draw bounding boxes if above confidence threshold
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            if score < conf_threshold:  # Skip low-confidence detections
                continue

            x1, y1, x2, y2 = map(int, box)  # Convert box to integers
            label = f"{class_names[cls]}: {score:.2f}"

            # Draw bounding box
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            img = cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 255, 0), 2)

    # Convert image back to PIL format for displaying
    st.image(img, caption="Predictions", use_column_width=True)
