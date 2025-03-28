import streamlit as st
import torch
import cv2
import numpy as np
import google.generativeai as genai
from ultralytics import YOLO
from PIL import Image
import pyttsx3

yolo_model = YOLO("yolov8l.pt") 
engine = pyttsx3.init()  # Initialize text-to-speech engine
GEMINI_API_KEY = "YOUR_KEY"
genai.configure(api_key=GEMINI_API_KEY)

def detect_objects(image):
    """Detects objects in an image using YOLOv8."""
    results = yolo_model(image)
    detected_objects = set()

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            detected_objects.add(class_name)

    return list(detected_objects)  # Convert set to list

def explain_with_gemini(object_name):
    """Generates an AI-based explanation and a fun fact using Gemini."""
    model = genai.GenerativeModel("gemini-1.5-pro")  
    response = model.generate_content(
        f"Explain what a '{object_name}' is in simple words. Then, give me one fun fact about it. "
        "Format the response like this: \nExplanation: <your explanation here>\nFun Fact: <your fun fact here>"
    )

    if response and response.text:
        text = response.text.strip()
        
        explanation = "No explanation available."
        fun_fact = "No fun fact found."

        if "Explanation:" in text and "Fun Fact:" in text:
            parts = text.split("Fun Fact:")
            explanation = parts[0].replace("Explanation:", "").strip()
            fun_fact = parts[1].strip() if len(parts) > 1 else "No fun fact found."
        
        return explanation, fun_fact

    return "No explanation available.", "No fun fact found."


def speak(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

#Streamlit UI
st.title("🔍 AI Object Identifier")
st.write("Upload an image, and the AI will detect objects and explain them.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    print("The program uses a large model but still it can make mistakes detecting multiple objects in an image.")

    # Convert image to OpenCV format for YOLO
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Detect objects
    objects = detect_objects(image_cv)
    st.write(f"📸 Objects Detected: **{', '.join(objects)}**")

    for obj in objects:
        explanation, fun_fact = explain_with_gemini(obj)
        st.subheader(f"🔍 {obj.capitalize()}")
        st.write(explanation)
        st.write(f"💡Fun Fact: {fun_fact if fun_fact else 'No fun fact found.'}")
        if st.button(f"🔊 Hear about {obj}"):
            speak(f"{obj}: {explanation}. Fun fact: {fun_fact}")