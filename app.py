import streamlit as st
from PIL import Image

try:
    from src.predict import load_detection_model, predict_image

    inference_import_error = None
except Exception as error:
    load_detection_model = None
    predict_image = None
    inference_import_error = error


st.set_page_config(
    page_title="Real/Fake Face Detection",
    page_icon="🔎",
)

st.title("Real/Fake Face Detection")
st.write("Upload a face image to receive a model prediction.")

if inference_import_error is not None:
    st.error(
        f"Inference module could not be imported: "
        f"{inference_import_error}"
    )


@st.cache_resource
def get_model():
    """Load the detection model once and reuse it."""
    if load_detection_model is None:
        raise RuntimeError("Inference module is unavailable")

    try:
        return load_detection_model()
    except Exception as error:
        raise RuntimeError(
            f"Detection model could not be loaded: {error}"
        ) from error


uploaded_file = st.file_uploader(
    "Choose a JPG or PNG image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        if inference_import_error is not None or predict_image is None:
            raise RuntimeError("Inference module is unavailable")

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_column_width=True,
        )

        # load_detection_model() returns:
        # model, serving signature, model path
        model, signature, model_path = get_model()

        # Reset the uploaded file position after Image.open()
        uploaded_file.seek(0)

        result = predict_image(
            uploaded_file,
            model=model,
            signature=signature,
            model_path=model_path,
        )

        st.subheader(result["predicted_class"])

        st.write(
            f"Model confidence: "
            f"{result['confidence']:.2%}"
        )

        st.caption(
            "This prediction can be wrong and is not proof "
            "of image authenticity."
        )

    except (
        OSError,
        ValueError,
        FileNotFoundError,
        RuntimeError,
    ) as error:
        st.error(f"Could not process this image: {error}")