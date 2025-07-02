import streamlit as st
import requests
from PIL import Image, ImageDraw
import io

# CONFIG
API_KEY = "eEdXUtNGshzr5pTDMeZ9"
MODEL_ID = "uae_dat_palm"
VERSION = "9"
ENDPOINT = f"https://detect.roboflow.com/{MODEL_ID}/{VERSION}?api_key={API_KEY}"

# UI
st.set_page_config(page_title="🌴 Palm Tree Detection", layout="centered")
st.title("🌴 UAE Date Palm Detection with Bounding Boxes")

uploaded_file = st.file_uploader("Upload satellite image to detect date palms", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("🔍 Sending image for detection...")

    response = requests.post(
        ENDPOINT,
        files={"file": uploaded_file.getvalue()}
    )

    if response.status_code == 200:
        result = response.json()
        predictions = result.get("predictions", [])

        if predictions:
            st.success(f"✅ {len(predictions)} objects detected")
            draw = ImageDraw.Draw(image)

            for obj in predictions:
                x = obj["x"]
                y = obj["y"]
                w = obj["width"]
                h = obj["height"]
                class_name = obj["class"]
                confidence = obj["confidence"]

                # Calculate box coords
                left = x - w / 2
                top = y - h / 2
                right = x + w / 2
                bottom = y + h / 2

                # Draw bounding box and label
                draw.rectangle([left, top, right, bottom], outline="magenta", width=3)
                draw.text((left, top - 10), f"{class_name} ({confidence:.2f})", fill="red")

            st.image(image, caption="🖼 Detected Results", use_column_width=True)
        else:
            st.warning("No objects detected in the image.")
    else:
        st.error(f"❌ Detection failed: {response.status_code} — {response.text}")
