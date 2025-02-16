from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os

app = Flask(__name__)
CORS(app)

class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=62):
        super(OptimizedCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
#Model loading
checkpoint = torch.load("Final_Model.pth", map_location=torch.device('cpu'))
model_state_dict = checkpoint['model_state_dict']

model = OptimizedCNN()
model.load_state_dict(model_state_dict)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

def predict(image):
  transform = transforms.Compose([
      transforms.Resize((128, 128)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5], std=[0.5])
  ])

  image = image.convert("L")
  input_tensor = transform(image).unsqueeze(0).to(device)

  # Perform prediction
  with torch.no_grad():
      output = model(input_tensor)
      _, predicted_class = torch.max(output, 1)

  print(f"The probability distribution:")
  print(output)
  print(f"Predicted Class Index: {predicted_class.item()}")

  predict_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A_caps': 10, 'B_caps': 11, 'C_caps': 12, 'D_caps': 13, 'E_caps': 14, 'F_caps': 15, 'G_caps': 16, 'H_caps': 17, 'I_caps': 18, 'J_caps': 19, 'K_caps': 20, 'L_caps': 21, 'M_caps': 22, 'N_caps': 23, 'O_caps': 24, 'P_caps': 25, 'Q_caps': 26, 'R_caps': 27, 'S_caps': 28, 'T_caps': 29, 'U_caps': 30, 'V_caps': 31, 'W_caps': 32, 'X_caps': 33, 'Y_caps': 34, 'Z_caps': 35, 'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45, 'k': 46, 'l': 47, 'm': 48, 'n': 49, 'o': 50, 'p': 51, 'q': 52, 'r': 53, 's': 54, 't': 55, 'u': 56, 'v': 57, 'w': 58, 'x': 59, 'y': 60, 'z': 61}

  label = list(predict_dict.keys())[predicted_class.item()]
  print(f"Predicted Label: {label}")
  return label


@app.route('/predict', methods=['POST'])
def check():
    try:
        #Receive Base64 image from frontend
        data = request.json['image']  # Get the image data from request

        #Convert Base64 to binary image data
        image_data = base64.b64decode(data.split(',')[1])  # Decode Base64

        #Convert to an actual image and save it
        image = Image.open(BytesIO(image_data))

        print("Image received successfully!")
        label = predict(image)
        return jsonify({'value': label})

    except Exception as e:
        return jsonify({'error': str(e)})
@app.route('/')
def index():
    return render_template("canvas.html")

@app.route('/about')
def about():
    return render_template("about.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) # Use Render's PORT or default to 5000
    app.run(host="0.0.0.0", port=port)

