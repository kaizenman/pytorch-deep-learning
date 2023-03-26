from torchvision import datasets
from torch.utils.data import dataloader

def create_dataloaders(train_dir, test_dir, transform, batch_size, num_workers=0):
  train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
  test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

  class_names = train_dataset.classes
  
  train_dataloader = dataloader.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=num_workers
  )
  test_dataloader = dataloader.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers
  )

  return train_dataloader, test_dataloader, class_names



