from flask import Flask, request, jsonify
from PIL import Image
import io
import torch

app = Flask(__name__)

# Global variable to hold your loaded PyTorch model
model = None

@app.route('/upload_model', methods=['POST'])
def upload_model():
    global model
    file = request.files.get('model')
    if not file:
        return jsonify({'error': 'No model uploaded'}), 400
    model = torch.load(file)
    model.eval()
    return jsonify({'message': f'Model {file.filename} loaded successfully'})

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 400

    prompt = request.form.get('prompt')
    image_file = request.files.get('image')

    if not prompt or not image_file:
        return jsonify({'error': 'Missing prompt or image'}), 400

    image = Image.open(image_file.stream).convert('RGB')

    # Preprocess image as your model expects
    # For example:
    # input_tensor = preprocess(image).unsqueeze(0)  # shape: [1, C, H, W]

    # Run model inference (example - modify as per your model)
    # output = model(input_tensor)

    # Create a meaningful response by combining prompt + output (this is your custom logic)
    # For demo, we'll just echo prompt and say image was processed
    result_text = f"Prompt: '{prompt}' processed with image analysis."
