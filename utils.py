import torch 
from pathlib import Path

import matplotlib.pyplot as plt
import os
from pathlib import Path

def visualize_learning(train_losses, test_losses, train_accuracies, test_accuracies, model_name):
  

  epochs = range(len(train_losses))
  
  f = plt.figure(figsize=(9, 9))
  plt.subplot(2, 2, 1)

  plt.plot(epochs, train_losses, label='Train Loss')
  plt.plot(epochs, test_losses, label='Test Loss')
  plt.title('Loss')
  plt.xlabel(xlabel='Epochs')
  plt.legend()

  plt.subplot(2, 2, 2)
  plt.plot(epochs, train_accuracies, label='Train Accuracy')
  plt.plot(epochs, test_accuracies, label='Test Accuracy')
  plt.title('Accuracy')
  plt.xlabel(xlabel='Epochs')
  plt.legend()


  results_dir = Path('results')
  results_image_name = model_name + '.png'
  results_image = results_dir / results_image_name 

  if not results_dir.is_dir():
    results_dir.mkdir(parents=True, exist_ok=True)

  f.savefig(results_image)
  plt.close(f)

def save_model(model, dir, model_name):
  target_dir = Path(dir)
  target_dir.mkdir(parents=True, exist_ok=True)
  model_save_path = Path(target_dir / model_name)
  print(f'Saving model to {model_save_path}...')
  torch.save(obj=model.state_dict(), f=model_save_path)
  print(f'Saved.')