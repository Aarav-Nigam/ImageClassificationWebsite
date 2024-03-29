import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import json
import matplotlib.pyplot as plt

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
    _, predicted = torch.max(outputs, 1)
    confidence = torch.softmax(outputs, dim=1)[0][predicted].item() * 100
    return predicted.item(), confidence

st.set_page_config(
    page_title="Image Classifier",
    page_icon=":robot_face:",
    layout="wide"
)
# Streamlit app
st.title("Image Classification \n(ResNet Architecture)")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Predict button
predict_button = st.button("Predict")

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    prediction, confidence = predict(image)
    class_name = labels[prediction]
    with col2:
        st.header("Prediction: "+class_name.upper() )
        st.header("Confidence: "+ f"{confidence:.2f}%")
        # Create a pie chart to show confidence
        fig, ax = plt.subplots()
        ax.pie([confidence, 100 - confidence], labels=["Confidence", "Other"], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
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