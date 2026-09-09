import streamlit as st
from PIL import Image

from src.predict import load_detection_model, predict_image

st.set_page_config(page_title="Real/Fake Face Detection", page_icon="🔎")
st.title("Real/Fake Face Detection")
st.write("Upload a face image to receive a model prediction.")


@st.cache_resource
def get_model():
    return load_detection_model()


uploaded_file = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)
        model, model_path = get_model()
        result = predict_image(uploaded_file, model=model, model_path=model_path)
        st.subheader(result["predicted_class"])
        st.write(f"Model confidence: {result['confidence']:.2%}")
        st.caption("This prediction can be wrong and is not proof of image authenticity.")
    except (OSError, ValueError, FileNotFoundError, RuntimeError) as error:
        st.error(f"Could not process this image: {error}")
