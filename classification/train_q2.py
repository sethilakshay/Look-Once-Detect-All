import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
import random


class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        # Loading pre_trained ResNet-18
        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')

        self.classes = num_classes
        self.flat_dim = self.resnet.fc.in_features

        # Creating a new fully connected layer
        self.fc = nn.Linear(self.flat_dim, num_classes)

        # Removing the last layer of the ResNet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])


    def forward(self, x):
        batch_size = x.shape[0]
        x = self.resnet(x)
        flat_x = x.view(batch_size, self.flat_dim)
        out = self.fc(flat_x)
        
        return out


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    args = ARGS(
        epochs=50,
        inp_size=224,
        use_cuda=True,
        val_every=70,
        lr=5e-5,
        batch_size=256,
        step_size=20,
        gamma=0.5,
        save_at_end = True,
        run_name="Q2_BestModel"
    )
    
    print(args)

    model = ResNet(len(VOCDataset.CLASS_NAMES)).to(args.device)

    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)
