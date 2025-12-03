import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import os
import glob


st.set_page_config(page_title="classifier", layout="centered")
st.title("Pill Defect Detector")


@st.cache_resource
def load_model():
    models_dir = "models/"
    model_files = glob.glob(os.path.join(models_dir, "*.pth"))

    if not model_files:
        return None, "No .pth models found in models/ directory"

    model_path = model_files[1]

    from torchvision.models import convnext_tiny

    model = convnext_tiny(pretrained=False)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)

    state_dict = torch.load(model_path, map_location=torch.device("mps"))
    model.load_state_dict(state_dict)
    model.eval()

    return model, model_path


transform = transforms.Compose(
    [
        transforms.Resize((225, 225)),
        transforms.ToTensor(),
    ]
)

class_names = ["bad", "good"]


model, model_info = load_model()

if model is None:
    st.error(model_info)
else:
    st.success(f"Model loaded: {model_info}")

    uploaded_files = st.file_uploader(
        "Upload medicine images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        cols = st.columns(min(len(uploaded_files), 3))

        for idx, uploaded_file in enumerate(uploaded_files):
            col = cols[idx % 3]

            image = Image.open(uploaded_file).convert("RGB")

            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_class].item()

            predicted_label = class_names[pred_class]

            with col:
                st.image(image, caption=uploaded_file.name, use_container_width=True)
                if predicted_label == "good":
                    st.success(f"✅ **GOOD** ({confidence:.1%})")
                else:
                    st.error(f"❌ **DEFECTIVE** ({confidence:.1%})")

                st.caption(
                    f"Defective: {probabilities[0][0]:.1%} | Good: {probabilities[0][1]:.1%}"
                )
