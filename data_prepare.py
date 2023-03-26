import os
import requests

from pathlib import Path
from zipfile import ZipFile

data_dir = Path('data')
images_dir = data_dir / 'pizza_steak_sushi'
zipfile_name = 'pizza_steak_sushi.zip'

if not images_dir.is_dir():
  print(f'{images_dir} directory does not exist. Creating...')
  images_dir.mkdir(parents=True, exist_ok=True)

with open(data_dir / zipfile_name, 'wb') as f:
  request = requests.get('https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip')  
  print(f'Downloading pizza_steak_sushi.zip...')
  f.write(request.content)

with ZipFile(data_dir / zipfile_name, 'r') as zipfile:
  print(f'Extracting {zipfile_name}...')
  zipfile.extractall(images_dir)
  os.remove(data_dir / zipfile_name)
  print(f'Done.')
