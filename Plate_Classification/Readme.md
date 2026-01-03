# Follow the steps below to run inference on a trained EfficientNet-B0 HSRP classifier using Google Colab.

### 1️⃣ Open Google Colab

Go to:
https://colab.research.google.com

Create a new Python notebook.

### 2️⃣ Enable GPU (Recommended)

In Colab menu:

Runtime → Change runtime type → Hardware accelerator → GPU → Save

### 3️⃣ Install Required Dependencies

### Run the following cell:

 ``` pip install torch torchvision pillow matplotlib ```

### 4️⃣ Upload Model Weights and Test Image

Upload the trained model file:

## efficientnet_b0_finetuned.pth


### 5️⃣ Run Inference (Single Image)

Copy and run the following single cell:

```
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

 #Device
device = "cuda" if torch.cuda.is_available() else "cpu"

 #Load model
model = models.efficientnet_b0(pretrained=False)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 1)

model.load_state_dict(
    torch.load("efficientnet_b0_finetuned.pth", map_location=device)
)

model = model.to(device)
model.eval()

 #Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

 #Load image
img_path = "/content/sample_plate.jpg"  # change to your image name
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

 #Inference
with torch.no_grad():
    logit = model(input_tensor)
    prob = torch.sigmoid(logit).item()

pred = "HSRP" if prob > 0.5 else "NON_HSRP"

print(f"Prediction : {pred}")
print(f"HSRP Probability : {prob:.4f}")

 #Display result
plt.imshow(image)

plt.axis("off")
plt.title(f"{pred} (p={prob:.2f})")
plt.show()
```
## this is important since efficientnet apllies on-the-fly transformations on the image
