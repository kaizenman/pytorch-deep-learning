# pytorch-deep-learning
Going modular

## Prepare data
```python3 data_prepare.py```

## Train
```
python3 train.py --training_dir 'data/pizza_steak_sushi/train' --testing_dir 'data/pizza_steak_sushi/test'  --lr 0.001 --num_epochs 5 --model_name 'model_jit.pth'
```
## Predict
Pizza:
```
python3 predict.py --model '/home/user/pytorch/pytorch-deep-learning/model/test.pth' --image 'data/pizza_steak_sushi/test/pizza/648055.jpg'
```
Sushi:
```
python3 predict.py --model '/home/user/pytorch/pytorch-deep-learning/model/bbb.pth' --image 'data/pizza_steak_sushi/test/sushi/3177743.jpg'
```