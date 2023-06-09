import argparse
import torch
import torchvision
import json

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--base', type=str, required=True)

args = parser.parse_args()

model_path = Path(args.model)
image_path = Path(args.image)

base=args.base

# Read the JSON data from the file
with open("classes.json", "r") as file:
    json_data = file.read()
class_names = json.loads(json_data)

if base == 'b0':
  transforms = torchvision.transforms.transforms.Compose([
    torchvision.transforms.transforms.Resize((228, 228)),
    torchvision.transforms.transforms.ToTensor(),
    torchvision.transforms.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  FEATURES=1280
  weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
  loaded_model = torchvision.models.efficientnet_b0(weights).to(device)
elif base == 'b2':
  transforms = torchvision.transforms.transforms.Compose([
    torchvision.transforms.transforms.Resize((288, 288)),
    torchvision.transforms.transforms.ToTensor(),
    torchvision.transforms.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  FEATURES=1408
  weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
  loaded_model = torchvision.models.efficientnet_b2(weights).to(device)

# freeze the model
for param in loaded_model.parameters():
  param.requires_grad = False
loaded_model.classifier = torch.nn.Sequential(
  torch.nn.Dropout(p=0.2, inplace=True),
  torch.nn.Linear(in_features=FEATURES, out_features=3, bias=True)
).to(device)

loaded_model.eval()
loaded_model.load_state_dict(torch.load(model_path))

with torch.inference_mode():
  img = Image.open(image_path)
  transformed_image = transforms(img)
  target_image_pred = loaded_model(transformed_image.unsqueeze(dim=0).to(device))
  displayed_image = transformed_image.permute(1, 2, 0).to('cpu')
  f = plt.figure(figsize=(9,9))
  plt.imshow(displayed_image)
  logit = torch.softmax(target_image_pred, dim=1).argmax(dim=1)
  plt.title(label={class_names[str(logit.item())]})
  plt.axis(False)
  f.savefig('predicted.png')
  plt.close(f)

  print(f'Predicted class: {class_names[str(logit.item())]}')