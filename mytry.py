import streamlit as st
import replicate
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch
from PIL import Image
import requests
from io import BytesIO

# Set up the Hugging Face model for image classification
image_model_name = "huggingface/medicinal_plants_image_detection"
image_model = AutoModelForImageClassification.from_pretrained(image_model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(image_model_name)

# Replicate API configuration for your fine-tuned text model
replicate_model_url = "https://api.replicate.com/v1/predictions"
replicate_model_version = "your-replicate-model-version"  # Replace with your model version
replicate_api_token = "your_replicate_api_token"  # Replace with your API token

# Function to call the Replicate API for text generation
def generate_text_response(prompt):
    headers = {
        "Authorization": f"Token {replicate_api_token}",
        "Content-Type": "application/json",
    }
    data = {
        "version": replicate_model_version,
        "input": {"prompt": prompt},
    }
    response = requests.post(replicate_model_url, headers=headers, json=data)
    response_json = response.json()
    return response_json.get("output", {}).get("text", "No response")

# Function to process the image and predict using Hugging Face model
def get_plant_prediction(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = image_model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = image_model.config.id2label[predicted_class_idx]
    return predicted_class

# Streamlit UI
st.title("Plant Chatbot")

st.sidebar.title("Upload Plant Image or Ask a Question")

user_input = st.sidebar.text_input("Ask a question about plants")

uploaded_file = st.sidebar.file_uploader("Upload an image of the plant")

if uploaded_file is not None:
    image = Image.open(BytesIO(uploaded_file.read()))
    plant_prediction = get_plant_prediction(image)
    if plant_prediction:
        st.write(f"Predicted plant or disease: {plant_prediction}")
        response = generate_text_response(f"Tell me about {plant_prediction}")
        st.write(f"Model says: {response}")

if user_input:
    response = generate_text_response(user_input)
    st.write(f"Model says: {response}")
