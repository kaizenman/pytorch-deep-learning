# pytorch-deep-learning
Pizza/Steak/Sushi classifier
## Prepare data
```python3 data_prepare.py```

## Train
```
python3 train.py --training_dir 'data/pizza_steak_sushi/train' --testing_dir 'data/pizza_steak_sushi/test'  --lr 0.001 --num_epochs 100 --model_name 'gpu_model_v3.pth'
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