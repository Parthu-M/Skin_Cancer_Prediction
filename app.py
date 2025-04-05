from flask import Flask, render_template, request
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import SwinForImageClassification, SwinConfig

app = Flask(__name__)

# Define class labels
label_mapping = {
    0: "Actinic Keratoses (akiec)",
    1: "Basal Cell Carcinoma (bcc)",
    2: "Benign Keratosis (bkl)",
    3: "Dermatofibroma (df)",
    4: "Melanoma (mel)",
    5: "Nevus (nv)",
    6: "Vascular Lesion (vasc)"
}

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = SwinConfig.from_pretrained("microsoft/swin-tiny-patch4-window7-224", num_labels=7)
model = SwinForImageClassification(config)
model.load_state_dict(torch.load("swin_transformer_skin_cancer.pth", map_location=device))
model.to(device)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    image_path = os.path.join("D:/Melanoma-Skin-Cancer-Detection-using-Swin-Transformer-main/uploads", file.filename)
    file.save(image_path)

    # Perform prediction
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image).logits
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]  # Get probability distribution
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class_name = label_mapping[predicted_class_idx]

    # Convert probabilities to readable format
    class_probabilities = {label_mapping[i]: f"{prob:.4f}" for i, prob in enumerate(probabilities.tolist())}

    # Pass result to template
    result = {
        "predicted_short": predicted_class_idx,
        "predicted_full": predicted_class_name,
        "class_probabilities": class_probabilities
    }

    return render_template("result.html", image_path=image_path, result=result)

if __name__ == '__main__':
    app.run(debug=True)