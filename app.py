from pathlib import Path

import streamlit as st
from PIL import Image

from src.predict import load_detection_model
from src.preprocessing import preprocess_image

st.set_page_config(page_title="Real/Fake Face Detection", page_icon="🔎")
st.title("Real/Fake Face Detection")
st.write("Upload a face image to receive a model prediction.")


@st.cache_resource
def get_model():
    return load_detection_model()[0]


uploaded_file = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)
        uploaded_file.seek(0)
        array = preprocess_image(uploaded_file)
        model = get_model()
        probability = float(model.predict(array[None, ...], verbose=0)[0][0])
        predicted_class = "REAL" if probability >= 0.5 else "FAKE"
        confidence = probability if predicted_class == "REAL" else 1.0 - probability
        st.subheader(predicted_class)
        st.write(f"Confidence: {confidence:.2%}")
    except (OSError, ValueError) as error:
        st.error(f"Could not process this image: {error}")
