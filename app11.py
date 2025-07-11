import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import time
from io import BytesIO
import requests

# Load the trained model
MODEL_PATH = "final_model112.h5"
model = load_model(MODEL_PATH)
 
# # Define class labels
# label_map = {
#     0: "Pigmented Benign Keratosis",
#     1: "Melanoma",
#     2: "Vascular Lesion",
#     3: "Actinic Keratosis",
#     4: "Squamous Cell Carcinoma",
#     5: "Basal Cell Carcinoma",
#     6: "Seborrheic Keratosis",-
#     7: "Dermatofibroma",
#     8: "Nevus",
# }

label_map = {
    0: "Actinic Keratosis",
    1: "Basal Cell Carcinoma",
    2: "benign keratosis",
    3: "Dermatofibroma",
    4: "melanocytic nevi",
    5: "pyogenic granulomas and hemorrhage",
    6: "melanoma",

}

# Streamlit App
def main():
    st.set_page_config(page_title="Skin Disease Classifier", page_icon="🩺")

    st.title("🩺 Skin Disease Classification Dashboard")
    st.markdown("Upload an image of your skin lesion or provide an image URL to classify it.")

    st.sidebar.header("Upload or Provide URL")

    # File uploader or URL input
    uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    image_url = st.sidebar.text_input("Or enter an image URL")

    img = None  # Initialize image variable

    if image_url:
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(img, caption="Fetched Image from URL", use_column_width=True)
        except Exception as e:
            st.error("Failed to fetch the image from the provided URL.")
            img = None

    elif uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if img and st.button("📊 Predict Skin Disease"):
        with st.spinner("Analyzing..."):
            start_time = time.time()

            # Preprocess image
            img = img.resize((100, 75))  # ✅ Fix image resizing
            img_array = np.array(img) / 255.0  # ✅ Normalize
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_class] * 100  # ✅ Fix confidence extraction

            end_time = time.time()

            # Display result
            st.success(f"🩻 **Prediction:** {label_map[predicted_class]}")
            # st.write(f"📊 **Confidence:** {confidence:.2f}%")
            st.write(f"🕒 **Prediction Time:** {end_time - start_time:.2f} seconds")

    else:
       st.info("Please upload an image or provide a URL to start the prediction.")

if __name__ == "__main__":
    main()
