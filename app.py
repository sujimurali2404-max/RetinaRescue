
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image
import numpy as np
import cv2
from pathlib import Path


# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="RetinaRescue AI",
    page_icon="👁️",
    layout="wide"
)

MODEL_PATH = Path(__file__).resolve().parent / "retinarescue.pth"


# ---------------- IMAGE ENHANCEMENT ----------------
def enhance_retinal_image(image):
    image_np = np.array(image.convert("RGB"))

    green = image_np[:, :, 1]

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_green = clahe.apply(green)

    enhanced = image_np.copy()
    enhanced[:, :, 1] = enhanced_green

    return Image.fromarray(enhanced)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.resolve()}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    class_names = checkpoint.get(
        "class_names",
        ["Healthy", "Diabetic Retinopathy"]
    )

    image_size = checkpoint.get("image_size", 224)

    model = mobilenet_v3_small(weights=None)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(
        in_features,
        len(class_names)
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, class_names, image_size


# ---------------- PREPROCESS IMAGE ----------------
def preprocess_image(image, image_size):
    enhanced_image = enhance_retinal_image(image)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(enhanced_image).unsqueeze(0)

    return enhanced_image, tensor


# ---------------- PREDICTION ----------------
def predict_image(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    predicted_class = class_names[predicted_index]

    confidence = (
        float(probabilities[predicted_index].item()) * 100
    )

    return predicted_class, confidence


# ---------------- KEY POINTS ----------------
def show_key_points(predicted_class):
    label = predicted_class.lower().strip()

    st.markdown("### Key Points")

    if (
        "diabetic retinopathy" in label
        or label in ["dr", "dr positive"]
    ):
        points = [
            "The model classified the image as Diabetic Retinopathy.",
            "Diabetic retinopathy can involve microaneurysms.",
            "Retinal haemorrhages and hard exudates may occur.",
            "Cotton-wool spots and abnormal new blood vessels may occur.",
            "The classifier does not identify the location of individual lesions."
        ]
    else:
        points = [
            "The model classified the image as Healthy.",
            "Image quality can affect the prediction.",
            "This classifier does not evaluate every eye condition.",
            "A Healthy prediction does not rule out all retinal disease."
        ]

    for point in points:
        st.markdown(f"- {point}")


# ---------------- MAIN APP ----------------
def main():
    st.title("👁️ RetinaRescue AI")
    st.subheader("Retinal Image Classification")

    st.markdown(
        "Upload a retinal image to view the original image, "
        "CLAHE-enhanced image, AI classification, and model score."
    )

    st.warning(
        "This is a research prototype, not a fully developed medical diagnostic tool. "
       
    )

    uploaded_file = st.file_uploader(
        "Upload a retinal image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("Upload a retinal image to begin.")
        return

    try:
        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

        model, class_names, image_size = load_model()

        enhanced_image, image_tensor = preprocess_image(
            original_image,
            image_size
        )

        predicted_class, confidence = predict_image(
            model,
            image_tensor,
            class_names
        )

        st.markdown("---")
        st.markdown("## Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Image")
            st.image(
                original_image,
                use_container_width=True
            )

        with col2:
            st.markdown("### CLAHE-Enhanced Image")
            st.image(
                enhanced_image,
                use_container_width=True
            )

        st.markdown("---")
        st.markdown("## AI Prediction")

        result_col, confidence_col = st.columns(2)

        with result_col:
            st.metric(
                label="Predicted Class",
                value=predicted_class
            )

        with confidence_col:
            st.metric(
                label="Model Confidence Score",
                value=f"{confidence:.2f}%"
            )

        st.progress(
            max(0.0, min(1.0, confidence / 100.0))
        )


        st.markdown("---")
        show_key_points(predicted_class)

    except FileNotFoundError as error:
        st.error(str(error))

    except Exception as error:
        st.error(f"An error occurred: {error}")


if __name__ == "__main__":
    main()
