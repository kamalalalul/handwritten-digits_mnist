# app.py
from fastapi import FastAPI, UploadFile, File
import uvicorn
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

# Define your model architecture (Net class)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

app = FastAPI()

# Load the model
model = Net()
model.load_state_dict(torch.load('mnist_model.pt'))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image = Image.open(io.BytesIO(await file.read()))
    # Apply transformations
    image = transform(image).unsqueeze(0)
    # Predict
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return {"prediction": predicted.item()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
