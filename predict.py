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
args = parser.parse_args()

model_path = Path(args.model)
image_path = Path(args.image)

# Read the JSON data from the file
with open("classes.json", "r") as file:
    json_data = file.read()
class_names = json.loads(json_data)


loaded_model = torchvision.models.efficientnet_b0().to(device)
# freeze the model
for param in loaded_model.parameters():
  param.requires_grad = False

loaded_model.classifier = torch.nn.Sequential(
  torch.nn.Dropout(p=0.2, inplace=True),
  torch.nn.Linear(in_features=1280, out_features=3, bias=True)
).to(device)

loaded_model.eval()
loaded_model.load_state_dict(torch.load(model_path))

with torch.inference_mode():
  img = Image.open(image_path)

  f = plt.figure(figsize=(9,9))
  plt.imshow(img)

  convert_tensor = torchvision.transforms.ToTensor()
  X = convert_tensor(img).to(device) 

  y_pred = loaded_model(X.unsqueeze(dim=0))

  print(torch.softmax(y_pred, dim=1))

  logit = torch.softmax(y_pred, dim=1).argmax(dim=1)

  plt.title(label={class_names[str(logit.item())]})
  plt.axis(False)
  f.savefig('predicted.png')
  plt.close(f)

  print(f'Predicted class: {class_names[str(logit.item())]}')