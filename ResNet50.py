import torch
import torch.nn as nn
from torchvision import models
from ImageClassificationBase import ImageClassificationBase

CLASSES=12

class ResNet50(ImageClassificationBase):
  def __init__(self):
    super().__init__()
    self.network = models.resnet50(weights='DEFAULT')
    num_features = self.network.fc.in_features
    self.network.fc = nn.Linear(num_features, CLASSES)

  def forward(self, xb):
    return torch.sigmoid(self.network(xb))