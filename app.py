
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="centered"
)

# Load trained model
MODEL_PATH = "diabetic_retinopathy_cnn.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names
class_names = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe"
]

# Title
st.title("👁️ Diabetic Retinopathy Detection")
st.write("Upload a retinal image to get a CNN-based prediction.")

st.info(
    "This application is a student/research project and "
    "is not a medical diagnosis."
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload a retinal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Retinal Image",
        use_container_width=True
    )

    # Prepare image
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction button
    if st.button("🔍 Predict"):

        prediction = model.predict(image_array, verbose=0)

        predicted_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_index]
        confidence = prediction[0][predicted_index] * 100

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.write(
            f"Confidence: {confidence:.2f}%"
        )

        st.warning(
            "For educational/research purposes only. "
            "Please consult a qualified medical professional "
            "for clinical evaluation."
        )
