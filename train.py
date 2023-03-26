import torch
import torchvision
import torchmetrics
import argparse

import data_setup, engine, utils

import torchinfo
import random

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--training_dir', type=str, required=True)
parser.add_argument('--testing_dir', type=str, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--num_epochs', type=int, required=True)
parser.add_argument('--model_name', type=str, required=True)
args = parser.parse_args()

model_save_dir = 'model'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

model = torchvision.models.efficientnet_b0(weights).to(device)

# freeze the model
for param in model.parameters():
  param.requires_grad = False

final_transform = torchvision.transforms.Compose([
  torchvision.transforms.Resize((228, 228)),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  )
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
  train_dir=args.training_dir,
  test_dir=args.testing_dir,
  transform=final_transform,
  batch_size=32
)

model.classifier = torch.nn.Sequential(
  torch.nn.Dropout(p=0.2, inplace=True),
  torch.nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
).to(device)

# torchinfo.summary(
#   model=model,
#   input_size=[32, 3, 224, 224],
#   col_names=["input_size", "output_size", "num_params", "trainable"],
#   col_width=20,
#   row_settings=["var_names"]
# )

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = torch.nn.CrossEntropyLoss().to(device)
accuracy_fn = torchmetrics.Accuracy(task='multiclass', num_classes=len(class_names)).to(device)


train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(args.num_epochs):
  train_loss, train_accuracy = engine.train_step(
    model, 
    train_dataloader, 
    optimizer,
    loss_fn,
    accuracy_fn,
    device
  )  
  
  test_loss, test_accuracy = engine.test_step(
    model,
    test_dataloader,
    loss_fn,
    accuracy_fn,
    device
  )

  train_losses.append(train_loss.item())
  train_accuracies.append(train_accuracy.item())
  test_losses.append(test_loss.item())
  test_accuracies.append(test_accuracy.item())

  print(f'epoch: {epoch} | train_loss: {train_loss:.4f} | train_accuracy: {train_accuracy:.4f} | test_loss: {test_loss:.4f} | test_accuracy: {test_accuracy:.4f}')

utils.write_class_names(class_names)
utils.visualize_learning(train_losses, test_losses, train_accuracies, test_accuracies, args.model_name)
utils.save_model(model, model_save_dir, args.model_name)
