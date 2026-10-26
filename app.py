
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

MODEL_PATH = Path("models/retinarescue.pth")


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

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index].item()) * 100

    return predicted_class, confidence


# ---------------- CIRCLE CANDIDATE REGIONS ----------------
def mark_candidate_regions(image):
    image_np = np.array(image.convert("RGB"))
    height, width = image_np.shape[:2]

    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    # Red-coloured candidate regions
    lower_red_1 = np.array([0, 45, 40])
    upper_red_1 = np.array([15, 255, 255])
    lower_red_2 = np.array([160, 45, 40])
    upper_red_2 = np.array([180, 255, 255])

    red_mask_1 = cv2.inRange(
        hsv, lower_red_1, upper_red_1
    )
    red_mask_2 = cv2.inRange(
        hsv, lower_red_2, upper_red_2
    )

    red_mask = cv2.bitwise_or(
        red_mask_1, red_mask_2
    )

    # Bright yellow candidate regions
    lower_yellow = np.array([15, 35, 100])
    upper_yellow = np.array([45, 255, 255])

    yellow_mask = cv2.inRange(
        hsv, lower_yellow, upper_yellow
    )

    combined_mask = cv2.bitwise_or(
        red_mask, yellow_mask
    )

    kernel = np.ones((3, 3), np.uint8)

    combined_mask = cv2.morphologyEx(
        combined_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        combined_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    marked_image = image_np.copy()
    count = 0

    min_area = max(5, int(height * width * 0.000005))
    max_area = max(
        min_area + 1,
        int(height * width * 0.003)
    )

    for contour in contours:
        area = cv2.contourArea(contour)

        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)

            center_x = x + w // 2
            center_y = y + h // 2
            radius = max(10, int(max(w, h) * 0.8))

            count += 1

            cv2.circle(
                marked_image,
                (center_x, center_y),
                radius,
                (255, 0, 0),
                2
            )

            cv2.putText(
                marked_image,
                str(count),
                (center_x, center_y - radius - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

    return Image.fromarray(marked_image), count


# ---------------- KEY POINTS ----------------
def show_key_points(predicted_class):
    label = predicted_class.lower()

    if (
        "diabetic retinopathy" in label
        or label.strip() in ["dr", "dr positive"]
    ):
        st.markdown("### Key Points")

        points = [
            "Microaneurysms",
            "Retinal haemorrhages",
            "Hard exudates",
            "Cotton-wool spots",
            "Abnormal new blood vessels"
        ]

        for point in points:
            st.markdown(f"- {point}")

    else:
        st.markdown("### Key Points")

        points = [
            "The model classified the uploaded image as Healthy.",
            "Image quality can affect the prediction.",
            "Other eye conditions are not evaluated by this classifier."
        ]

        for point in points:
            st.markdown(f"- {point}")


# ---------------- MAIN APP ----------------
def main():
    st.title("👁️ RetinaRescue AI")
    st.subheader("Retinal Image Classification")

    st.markdown(
        "Upload a retinal image to view the AI classification, "
        "confidence score, enhanced image, and highlighted candidate regions."
    )

    uploaded_file = st.file_uploader(
        "Upload a retinal image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("Upload a retinal image to begin.")
        return

    try:
        original_image = Image.open(uploaded_file).convert("RGB")

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

        marked_image, candidate_count = mark_candidate_regions(
            original_image
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
            st.markdown("### Enhanced Image")
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
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

        st.progress(
            max(0.0, min(1.0, confidence / 100.0))
        )

        st.markdown("---")
        st.markdown("## Highlighted Candidate Regions")

        st.image(
            marked_image,
            use_container_width=True
        )

        st.write(
            f"Candidate regions highlighted: {candidate_count}"
        )

        st.markdown("---")
        show_key_points(predicted_class)

    except FileNotFoundError as error:
        st.error(str(error))

    except Exception as error:
        st.error(f"An error occurred: {error}")


if __name__ == "__main__":
    main()
