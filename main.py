from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from torchvision import transforms
import torch
from PIL import Image
from ResNet50 import ResNet50

app = Flask(__name__)

# Enable CORS
CORS(app)

# Load the model
model = ResNet50()
model.load_state_dict(torch.load('./model.pth', map_location=torch.device('cpu')))
model.eval()
classes = ['battery', 'biological','brown-glass','cardboard', 'clothes','green-glass','metal','paper','plastic','shoes','trash','white-glass']
    

def get_prediction_class(prediction):
    return classes[prediction].capitalize()

def get_bin(prediction):
    if prediction in [2, 3, 5, 6,7,8, 11]:
        return 'Recycling'
    elif prediction in [4, 9, 10]:
        return "Garbage"
    elif prediction == 0:
        return "Battery"
    else:
        return "Organic"
    

# Preprocess the image
def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize((255, 255)),
                                        transforms.ToTensor()])
    return my_transforms(image).unsqueeze(0)

# Get the prediction
def get_prediction(image_tensor):
    with torch.no_grad():
        pred = model(image_tensor)
        _, predicted = torch.max(pred, 1)
        return predicted.item()

@app.route('/', methods=['GET'])
def hello():
    return "Hello World!"

# Define the POST request
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img = Image.open(file.stream)
        transformed_img = transform_image(img)
        prediction = get_prediction(transformed_img)
        pred_class = get_prediction_class(prediction)
        bin = get_bin(prediction)
        return jsonify({'prediction': pred_class, 'bin': bin})


if __name__ == '__main__':
    app.run(debug=True)
