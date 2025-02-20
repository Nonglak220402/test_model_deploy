import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("best.pt")  # Ensure best.pt is in the same directory

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

    # Convert PIL image to NumPy array (RGB format)
    img = np.array(image)

    # Convert to BGR format for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # YOLO Prediction
    results = model(img)

    # Class names
    class_names = [
        "Boo Pad Pongali",    # 0: Stir-fried Crab in Yellow Curry
        "Fried Fish Cakes",   # 1: Tod Mun Pla
        "Braised Pork",       # 2: Moo Palo
        "Stir-fried Kale",    # 3: Pad Kana
        "Gaeng Jued",         # 4: Clear Soup with Vegetables and Tofu
        "Grilled Chicken",    # 5: Gai Yang
        "Grilled Shrimp",     # 6: Kung Pao
        "Khao Kluk Kapi",     # 7: Shrimp Paste Fried Rice
        "Kuay Chap",          # 8: Thai-Chinese Noodle Soup
        "Yum Woon Sen"        # 9: Spicy Glass Noodle Salad
    ]

    detected_classes = []  # Store detected class names

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy().astype(int)  # Class indices

        # If no detections
        if len(boxes) == 0:
            st.write("⚠️ No detections found. Try another image.")
            continue

        # Draw bounding boxes if above confidence threshold
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            if score < conf_threshold:  # Skip low-confidence detections
                continue

            x1, y1, x2, y2 = map(int, box)  # Convert box to integers
            label = f"{class_names[cls]}: {score:.2f}"
            detected_classes.append(class_names[cls])  # Store detected class name

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

    # Convert back to RGB for Streamlit
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display result
    st.image(img, caption="Predictions", use_column_width=True)

    # Display detected class names
    if detected_classes:
        st.write("### 🏷️ Predicted Classes:")
        for cls in detected_classes:
            st.write(f"- {cls}")
