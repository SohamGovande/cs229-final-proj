import torch.nn as nn
import torchvision

class DenseNet(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNet, self).__init__()
        self.model = torchvision.models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)