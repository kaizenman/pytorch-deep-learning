import torch
import torchvision
import torchmetrics
import argparse

import model_setup, data_setup, engine, utils

import torchinfo

parser = argparse.ArgumentParser()
parser.add_argument('--training_dir', type=str, required=True)
parser.add_argument('--testing_dir', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--hidden_layers', type=int, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--num_epochs', type=int, required=True)
parser.add_argument('--model_name', type=str, required=True)
args = parser.parse_args()

model_save_dir = 'model'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transforms = torchvision.transforms.Compose([
  torchvision.transforms.Resize((64, 64)),
  torchvision.transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
  train_dir=args.training_dir,
  test_dir=args.testing_dir,
  transform=transforms,
  batch_size=args.batch_size
)

model = model_setup.TinyVGG(
  inputs=3,
  hidden=args.hidden_layers,
  outputs=len(class_names),
  device=device
).to(device)

#torchinfo.summary(model, [10, 3, 64, 64])
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = torch.nn.CrossEntropyLoss().to(device)
accuracy_fn = torchmetrics.Accuracy(task='multiclass', num_classes=len(class_names)).to(device)

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

  print(f'epoch: {epoch} | train_loss: {train_loss:.4f} | train_accuracy: {train_accuracy:.4f} | test_loss: {test_loss:.4f} | test_accuracy: {test_accuracy:.4f}')

utils.save_model(model, model_save_dir, args.model_name)
