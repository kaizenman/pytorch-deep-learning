import random
import argparse
import torch
import torch.utils.tensorboard
import torchvision
import torchmetrics
import torchinfo
from summary_writer import summary_writer

import data_prepare, data_setup, engine, utils

from pathlib import Path

def train(
  model,
  train_dataloader,
  test_dataloader,
  optimizer,
  loss_fn,
  accuracy_fn,
  epochs,
  model_name,
  device,
  writer
):
  results = {'train_loss': [],
             'train_accuracy': [],
             'test_loss': [],
             'test_accuracy': []}

  for epoch in range(epochs):
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

    results['train_loss'].append(train_loss.item())
    results['train_accuracy'].append(train_accuracy.item())
    results['test_loss'].append(test_loss.item())
    results['test_accuracy'].append(test_accuracy.item())

    print(f'epoch: {epoch} | train_loss: {train_loss:.4f} | train_accuracy: {train_accuracy:.4f} | test_loss: {test_loss:.4f} | test_accuracy: {test_accuracy:.4f}')



    if writer:
      writer.add_scalars(main_tag='Loss', tag_scalar_dict={'train_loss': train_loss, 
                                                           'test_loss': test_loss},
                      global_step=epoch)
      writer.add_scalars(main_tag='Accuracy', tag_scalar_dict={'train_accuracy': train_accuracy, 
                                                               'train_accuracy': test_accuracy},
                      global_step=epoch)
      writer.add_graph(model=model, input_to_model=torch.randn(32, 3, 224, 224).to(device))
      writer.close()

  utils.write_class_names(class_names)
  utils.visualize_learning(
    results['train_loss'],
    results['test_loss'],
    results['train_accuracy'],
    results['test_accuracy'],
    model_name
  )
  utils.save_model(model, model_save_dir, model_name)

def transfer_learning(model, features, transforms, train_dir, test_dir, batch_size=32):
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir,
    test_dir,
    transforms,
    batch_size
  )
  for param in model.parameters():
    param.requires_grad = False
  model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=features, out_features=len(class_names), bias=True)
  ).to(device)

  return model, train_dataloader, test_dataloader, class_names

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--skip_download', type=bool, required=False)
#parser.add_argument('--num_epochs', type=int, required=True)
args = parser.parse_args()

BATCH_SIZE=32
LEARNING_RATE=0.001
data_10_percent_src='https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip'
data_10_percent_dst='pizza_steak_sushi'
data_20_percent_src='https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip'
data_20_percent_dst='pizza_steak_sushi_20_percent'

if not args.skip_download:
  data_10_percent_path = data_prepare.download_data(data_10_percent_src, data_10_percent_dst)
  data_20_percent_path = data_prepare.download_data(data_20_percent_src, data_20_percent_dst)
else:
  print(f'skipping data download')
  data_10_percent_path = Path('data/pizza_steak_sushi')
  data_20_percent_path = Path('data/pizza_steak_sushi_20_percent')

train_dir_10_percent = data_10_percent_path / 'train'
train_dir_20_percent = data_20_percent_path / 'train'
test_dir_10_percent = data_10_percent_path / 'test'
test_dir_20_percent = data_20_percent_path / 'test'

model_save_dir = 'model'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

b0_features = 1280

b0_manual_transforms = torchvision.transforms.transforms.Compose([
  torchvision.transforms.transforms.Resize((228, 228)),
  torchvision.transforms.transforms.ToTensor(),
  torchvision.transforms.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

b2_manual_transforms = torchvision.transforms.transforms.Compose([
  torchvision.transforms.transforms.Resize((288, 288)),
  torchvision.transforms.transforms.ToTensor(),
  torchvision.transforms.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

B0_FEATURES=1280
B2_FEATURES=1408

effnetb0_10_model, effnetb0_10_train_dataloader, effnetb0_10_test_dataloader, class_names = transfer_learning(
  model=torchvision.models.efficientnet_b0(torchvision.models.EfficientNet_B0_Weights.DEFAULT).to(device),
  features=B0_FEATURES,
  transforms=b0_manual_transforms,
  train_dir=train_dir_10_percent,
  test_dir=test_dir_10_percent
)
effnetb0_20_model, effnetb0_20_train_dataloader, effnetb0_20_test_dataloader, class_names = transfer_learning(
  model=torchvision.models.efficientnet_b0(torchvision.models.EfficientNet_B0_Weights.DEFAULT).to(device),
  features=B0_FEATURES,
  transforms=b0_manual_transforms,
  train_dir=train_dir_20_percent,
  test_dir=test_dir_20_percent
)
effnetb2_10_model, effnetb2_10_train_dataloader, effnetb2_10_test_dataloader, class_names = transfer_learning(
  model=torchvision.models.efficientnet_b2(torchvision.models.EfficientNet_B2_Weights.DEFAULT).to(device),
  features=B2_FEATURES,
  transforms=b2_manual_transforms,
  train_dir=train_dir_10_percent,
  test_dir=test_dir_10_percent
)
effnetb2_20_model, effnetb2_20_train_dataloader, effnetb2_20_test_dataloader, class_names = transfer_learning(
  model=torchvision.models.efficientnet_b2(torchvision.models.EfficientNet_B2_Weights.DEFAULT).to(device),
  features=B2_FEATURES,
  transforms=b2_manual_transforms,
  train_dir=train_dir_20_percent,
  test_dir=test_dir_20_percent
)

print(f'Training on device: {device}')
#--------------------------------- EXPERIMENTS ---------------------------------------
# Experiment 1
# Pizza, Steak, Sushi 10% percent	EfficientNetB0	5
print(f'Experiment 1 | Pizza, Steak, Sushi 10% percent	EfficientNetB0 5')
train(
  model=effnetb0_10_model,
  train_dataloader=effnetb0_10_train_dataloader,
  test_dataloader=effnetb0_10_test_dataloader,
  optimizer=torch.optim.Adam(effnetb0_10_model.parameters(), lr=LEARNING_RATE),
  loss_fn=torch.nn.CrossEntropyLoss().to(device),
  accuracy_fn=torchmetrics.Accuracy(task='multiclass', num_classes=len(class_names)).to(device),
  epochs=5,
  model_name='b0_10_percent_5',
  device=device,
  writer=summary_writer('1 | dataset: pizza_steak_sushi | 10% percent	| model: EfficientNetB0 | epochs: 5', 'b0_10_percent_5')
)

# Experiment 2
print(f'Experiment 2 | Pizza, Steak, Sushi 10% percent	EfficientNetB2	5')
train(
  model=effnetb2_10_model,
  train_dataloader=effnetb2_10_train_dataloader,
  test_dataloader=effnetb2_10_test_dataloader,
  optimizer=torch.optim.Adam(effnetb2_10_model.parameters(), lr=LEARNING_RATE),
  loss_fn=torch.nn.CrossEntropyLoss().to(device),
  accuracy_fn=torchmetrics.Accuracy(task='multiclass', num_classes=len(class_names)).to(device),
  epochs=5,
  model_name='b2_10_percent_5',
  device=device,
  writer=summary_writer('2 | dataset: pizza_steak_sushi | 10% percent	| model: EfficientNetB2 | epochs: 5', 'b2_10_percent_5')
)

# Experiment 3
print(f'Experiment 3 | Pizza, Steak, Sushi 10% percent	EfficientNetB0	10')
train(
  model=effnetb0_10_model,
  train_dataloader=effnetb0_10_train_dataloader,
  test_dataloader=effnetb0_10_test_dataloader,
  optimizer=torch.optim.Adam(effnetb0_10_model.parameters(), lr=LEARNING_RATE),
  loss_fn=torch.nn.CrossEntropyLoss().to(device),
  accuracy_fn=torchmetrics.Accuracy(task='multiclass', num_classes=len(class_names)).to(device),
  epochs=10,
  model_name='b0_10_percent_10',
  device=device,
  writer=summary_writer('3 | dataset: pizza_steak_sushi | 10% percent	| model: EfficientNetB0 | epochs: 10', 'b0_10_percent_10')
)

# Experiment 4
print(f'Experiment 4 | Pizza, Steak, Sushi 10% percent	EfficientNetB2	10') 
train(
  model=effnetb2_10_model,
  train_dataloader=effnetb2_10_train_dataloader,
  test_dataloader=effnetb2_10_test_dataloader,
  optimizer=torch.optim.Adam(effnetb2_10_model.parameters(), lr=LEARNING_RATE),
  loss_fn=torch.nn.CrossEntropyLoss().to(device),
  accuracy_fn=torchmetrics.Accuracy(task='multiclass', num_classes=len(class_names)).to(device),
  epochs=10,
  model_name='b2_10_percent_10',
  device=device,
  writer=summary_writer('4 | dataset: pizza_steak_sushi | 10% percent	| model: EfficientNetB2 | epochs: 10', 'b2_10_percent_10')
)

# Experiment 5
print(f'Experiment 5 | Pizza, Steak, Sushi 20% percent	EfficientNetB0	5')
train(
  model=effnetb0_20_model,
  train_dataloader=effnetb0_20_train_dataloader,
  test_dataloader=effnetb0_20_test_dataloader,
  optimizer=torch.optim.Adam(effnetb0_20_model.parameters(), lr=LEARNING_RATE),
  loss_fn=torch.nn.CrossEntropyLoss().to(device),
  accuracy_fn=torchmetrics.Accuracy(task='multiclass', num_classes=len(class_names)).to(device),
  epochs=5,
  model_name='b0_20_percent_5',
  device=device,
  writer=summary_writer('5 | dataset: pizza_steak_sushi | 20% percent	| model: EfficientNetB0 | epochs: 5', 'b0_20_percent_5')
)

# Experiment 6
print(f'Experiment 6 | Pizza, Steak, Sushi 20% percent	EfficientNetB2	5')
train(
  model=effnetb2_20_model,
  train_dataloader=effnetb2_20_train_dataloader,
  test_dataloader=effnetb2_20_test_dataloader,
  optimizer=torch.optim.Adam(effnetb2_20_model.parameters(), lr=LEARNING_RATE),
  loss_fn=torch.nn.CrossEntropyLoss().to(device),
  accuracy_fn=torchmetrics.Accuracy(task='multiclass', num_classes=len(class_names)).to(device),
  epochs=5,
  model_name='b2_20_percent_5',
  device=device,
  writer=summary_writer('6 | dataset: pizza_steak_sushi | 20% percent	| model: EfficientNetB2 | epochs: 5', 'b2_20_percent_5')
)

# Experiment 7
print(f'Experiment 7 | Pizza, Steak, Sushi 20% percent	EfficientNetB0	10')
train(
  model=effnetb0_20_model,
  train_dataloader=effnetb0_20_train_dataloader,
  test_dataloader=effnetb0_20_test_dataloader,
  optimizer=torch.optim.Adam(effnetb0_20_model.parameters(), lr=LEARNING_RATE),
  loss_fn=torch.nn.CrossEntropyLoss().to(device),
  accuracy_fn=torchmetrics.Accuracy(task='multiclass', num_classes=len(class_names)).to(device),
  epochs=10,
  model_name='b0_20_percent_10',
  device=device,
  writer=summary_writer('7 | dataset: pizza_steak_sushi | 20% percent	| model: EfficientNetB0 | epochs: 10', 'b0_20_percent_10')
)

# Experiment 8
print(f'Experiment 8 | Pizza, Steak, Sushi 20% percent	EfficientNetB2	10')
train(
  model=effnetb2_20_model,
  train_dataloader=effnetb2_20_train_dataloader,
  test_dataloader=effnetb2_20_test_dataloader,
  optimizer=torch.optim.Adam(effnetb2_20_model.parameters(), lr=LEARNING_RATE),
  loss_fn=torch.nn.CrossEntropyLoss().to(device),
  accuracy_fn=torchmetrics.Accuracy(task='multiclass', num_classes=len(class_names)).to(device),
  epochs=10,
  model_name='b2_20_percent_10',
  device=device,
  writer=summary_writer('8 | dataset: pizza_steak_sushi | 20% percent	| model: EfficientNetB2 | epochs: 10', 'b2_20_percent_10')
)