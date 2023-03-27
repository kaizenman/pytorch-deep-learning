# pytorch-deep-learning
Pizza/Steak/Sushi classifier
## Prepare data
```python3 data_prepare.py```

## Run train experiments
```
python3 train.py
```

## Tensorboard
```
Ctrl+Shift+P -> Launch Tensorboard
```

## Predict
Pizza:
```
python3 predict.py --model '/home/user/pytorch/pytorch-deep-learning/model/test.pth' --image 'data/pizza_steak_sushi/test/pizza/648055.jpg'
```
Sushi:
```
python3 predict.py --model '/home/user/pytorch/pytorch-deep-learning/model/gpu_model_v3.pth' --image 'data/pizza_steak_sushi/test/sushi/1742201.jpg'
```
Steak
```
python3 predict.py --model '/home/user/pytorch/pytorch-deep-learning/model/gpu_model_v3.pth' --image 'data/pizza_steak_sushi/test/steak/502076.jpg'
```