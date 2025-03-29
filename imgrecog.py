import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model (pretrained)
model = YOLO("yolov8l.pt")

# Function to perform object detection
def detect_objects(image):
    results = model(image)
    objects = []
    for result in results:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = box
            label = model.names[int(class_id)]
            objects.append({
                "label": label,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(confidence * 100, 2)
            })
    return objects

# Function to draw bounding boxes
def draw_boxes(image, objects):
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        label = obj["label"]
        confidence = obj["confidence"]

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add label text
        text = f"{label} ({confidence}%)"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Streamlit UI
st.title("AI Object Detection & Annotation Tool")

# Sidebar options
option = st.sidebar.radio("Choose Input Source", ["Upload Image", "Real-time Camera"])

# Image Upload Section
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Detect objects
        detected_objects = detect_objects(image_np)

        # Editable object labels
        updated_objects = []
        if detected_objects:
            st.subheader("🔍 Detected Objects (Edit Labels if Needed)")
            for i, obj in enumerate(detected_objects):
                new_label = st.text_input(f"Edit label for {obj['label']}:", obj['label'], key=f"label_{i}")
                obj['label'] = new_label
                updated_objects.append(obj)

            # Draw bounding boxes on image
            annotated_image = draw_boxes(image_np.copy(), updated_objects)

            # Display annotated image
            st.image(annotated_image, caption="Annotated Image", use_container_width=True)

# Real-Time Camera Section
elif option == "Real-time Camera":
    st.subheader("📷 Live Camera Object Detection")

    # Initialize session state for camera control
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    # Start/Stop Camera Button
    if not st.session_state.camera_active:
        if st.button("Start Camera", key="start_camera"):
            st.session_state.camera_active = True
    else:
        if st.button("Stop Camera", key="stop_camera"):
            st.session_state.camera_active = False

    # If camera is active, start video stream
    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_objects = detect_objects(frame_rgb)
            annotated_frame = draw_boxes(frame.copy(), detected_objects)

            frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

        cap.release()
        cv2.destroyAllWindows()
