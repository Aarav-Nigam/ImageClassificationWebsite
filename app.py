import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import json
import matplotlib.pyplot as plt
import numpy as np

# Load pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()

# Load class labels from local JSON file
with open('label.json', 'r') as f:
    labels = json.load(f)

# Define transformations for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to make predictions
def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.topk(outputs, 5)
    confidences = torch.softmax(outputs, dim=1)[0][predicted[0]].tolist()
    predictions = [(labels[predicted[0][i].item()], confidences[i] * 100) for i in range(5)]
    return predictions

# Function to plot pie chart
def plot_pie_chart(predictions):
    other_confidence = 100 - sum([confidence for _, confidence in predictions])
    prediction_labels = [class_name.upper() for class_name, _ in predictions] + ['Other']
    prediction_confidences = [confidence for _, confidence in predictions] + [other_confidence]

    fig, ax = plt.subplots()
    ax.pie(prediction_confidences, labels=prediction_labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title("Top 5 Predictions with Respect to Others")
    return fig

st.set_page_config(
    page_title="Image Classifier",
    page_icon=":robot_face:",
    layout="wide"
)

# Background image
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D") center center;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Streamlit app
st.title("Image Classification \n(ResNet Architecture)")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Predict button
predict_button = st.button("Predict")

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    predictions = predict(image)
    with col2:
        st.header("Top 5 Predictions:")
        for i, (class_name, confidence) in enumerate(predictions):
            st.subheader(f"{i+1}. {class_name.upper()}")
            st.write(f"Confidence: {confidence:.2f}%")
            st.write("---")

    # Plot the pie chart
    fig = plot_pie_chart(predictions)
    st.pyplot(fig)

# Define the footer HTML content with CSS for sticky positioning
footer = '''
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
}
</style>
<div class="footer">
    <p>Developed  by <a style='display: block; text-align: center; color:white ;font-weight: bold; font-size:20px' href="https://portfolio-aarav.netlify.app/" target="_blank">~Aarav Nigam</a></p>
</div>
'''

# Display the footer using st.markdown
st.markdown(footer, unsafe_allow_html=True)
