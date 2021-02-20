import torch
import torch.nn as nn
import timm
import torchvision

class HornetsClassifier(nn.Module):
    def __init__(self, model_name, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # self.model = torchvision.models.resnet18(pretrained=True)

        for param in self.model.parameters():
            param.required_grad = False

        # num_features = self.model.fc.in_features
        # Replace last layer
        # self.model.fc = nn.Linear(num_features, n_class)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


# test_dataset = HornetsDataset(
#     img_path=test_path,
#     transform=transformdata.get_test_transforms()
# )
