import streamlit as st
import torch
import random
import os

from PIL import Image
from torchvision import transforms, models

#title
st.title("Wheat Crop Digital Twin")

#sensor data (dummy)
temperature = random.randint(20, 40)
humidity = random.randint(50, 95)

st.metric("Temperature", f"{temperature} °C")
st.metric("Humidity", f"{humidity}%")

#image_loading(later live image)
image_folder = "images"

image_files = [
    file for file in os.listdir(image_folder)
    if file.endswith((".jpg", ".png", ".jpeg"))
]

random_image = random.choice(image_files)

image_path = os.path.join(image_folder, random_image)

image = Image.open(image_path)

st.image(image, caption=random_image)

#model load

import torch
import torch.nn as nn
from torchvision import models

model = models.resnet50(weights=None)

model.fc = nn.Linear(model.fc.in_features, 11)

model.load_state_dict(
    torch.load(r"models\resnet50.pth", map_location="cpu")
)

model.eval()

#clase names
classes = [
    "Aphid",
    "Black_Rust",
    "Brown_Rust",
    "Crown and Root Rot",
    "Fusarium_Head_Blight",
    "Healthy","Leaf_Blight","Mite",
    "Septoria","Stem Fly","Stripe_Rust"
]
#process
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

input_tensor = transform(image)

input_tensor = input_tensor.unsqueeze(0)

#preds

with torch.no_grad():

    output = model(input_tensor)

    prediction = torch.argmax(output, dim=1)

predicted_class = classes[prediction.item()]

probabilities = torch.softmax(output, dim=1)

confidence = probabilities[0][prediction.item()] * 100

st.success(
    f"Prediction: {predicted_class}"
)

st.write(
    f"Confidence: {confidence:.2f}%"
)