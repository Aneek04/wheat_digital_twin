import streamlit as st
import torch
import torch.nn as nn
import random
import os
import gc

from PIL import Image
from torchvision import transforms, models

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Wheat Crop Digital Twin",
    layout="centered"
)

st.title("🌾 Wheat Crop Digital Twin")

# ---------------------------------------------------
# SENSOR DATA (DUMMY FOR NOW)
# ---------------------------------------------------

temperature = random.randint(20, 40)
humidity = random.randint(50, 95)

col1, col2 = st.columns(2)

with col1:
    st.metric("Temperature", f"{temperature} °C")

with col2:
    st.metric("Humidity", f"{humidity}%")

# ---------------------------------------------------
# LOAD RANDOM IMAGE
# ---------------------------------------------------

image_folder = "images"

image_files = [
    file for file in os.listdir(image_folder)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]

random_image = random.choice(image_files)

image_path = os.path.join(image_folder, random_image)

image = Image.open(image_path).convert("RGB")

st.image(
    image,
    caption=f"Selected Image: {random_image}",
    use_container_width=True
)

# ---------------------------------------------------
# CLASS NAMES
# ---------------------------------------------------

classes = [
    "Aphid",
    "Black_Rust",
    "Brown_Rust",
    "Crown_and_Root_Rot",
    "Fusarium_Head_Blight",
    "Healthy",
    "Leaf_Blight",
    "Mite",
    "Septoria",
    "Stem_Fly",
    "Stripe_Rust"
]

# ---------------------------------------------------
# IMAGE TRANSFORM
# ---------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------------------------------------
# LOAD MODEL ONLY ONCE (IMPORTANT)
# ---------------------------------------------------

@st.cache_resource
def load_model():

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(model.fc.in_features, 11)

    model.load_state_dict(
        torch.load(
            "models/resnet50.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model

model = load_model()

# ---------------------------------------------------
# PREPROCESS IMAGE
# ---------------------------------------------------

input_tensor = transform(image)

input_tensor = input_tensor.unsqueeze(0)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

with torch.no_grad():

    output = model(input_tensor)

    prediction = torch.argmax(output, dim=1)

    probabilities = torch.softmax(output, dim=1)

predicted_class = classes[prediction.item()]

confidence = probabilities[0][prediction.item()].item() * 100

# ---------------------------------------------------
# SHOW RESULTS
# ---------------------------------------------------

st.subheader(" Disease Prediction")

st.success(f"Prediction: {predicted_class}")

st.write(f"Confidence: {confidence:.2f}%")

# ---------------------------------------------------
# RISK ANALYSIS
# ---------------------------------------------------

risk = "LOW"

if humidity > 80 and predicted_class != "Healthy":
    risk = "HIGH"

elif humidity > 65:
    risk = "MEDIUM"

st.subheader("Disease Spread Risk")

if risk == "HIGH":
    st.error("HIGH RISK")

elif risk == "MEDIUM":
    st.warning("MEDIUM RISK")

else:
    st.success("LOW RISK")

# ---------------------------------------------------
# SYSTEM STATUS
# ---------------------------------------------------



st.write(f"Current Image: {random_image}")
st.write(f"Temperature: {temperature} °C")
st.write(f"Humidity: {humidity}%")
st.write(f"Disease Status: {predicted_class}")

# ---------------------------------------------------
# MEMORY CLEANUP
# ---------------------------------------------------

del input_tensor
del output
del probabilities

gc.collect()