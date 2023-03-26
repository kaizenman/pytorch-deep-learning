from torch import nn

class TinyVGG(nn.Module):
  def __init__(self, inputs, hidden, outputs, device):
    super().__init__()

    self.block_1 = nn.Sequential(
      nn.Conv2d(in_channels=inputs, out_channels=hidden, kernel_size=3),
      nn.ReLU(),
      nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=3),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.block_2 = nn.Sequential(
      nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=3),
      nn.ReLU(),
      nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=3),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=1)
    )
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features=25*25*hidden, out_features=outputs)
    )

  def forward(self, x):
    return self.classifier(self.block_2(self.block_1(x)))