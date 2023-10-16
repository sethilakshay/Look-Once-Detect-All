import torch
import torch.nn as nn
from train_q2 import ResNet
import numpy as np
import matplotlib.pyplot as plt
from voc_dataset import VOCDataset
from sklearn.manifold import TSNE


class ResNet_Repr(nn.Module):
    
    def __init__(self, modelPath):
        super().__init__()
        
        temp = torch.load(modelPath)
        temp = nn.Sequential(*list(temp.children())[:-1])
        
        self.newModel = temp
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.newModel(x)
        x = x.view(batch_size, -1)
        return x
    
def getRepresentations(model, testLoader):
    cntr = 0
    labels_list, features_list = [], []

    for data, target, wgt in testLoader:
        data, target, wgt = data.to(DEVICE), target.to(DEVICE), wgt.to(DEVICE)
        if cntr == 10:
            break

        labels_list.extend(target.detach().cpu().numpy())

        out = model(data.to(DEVICE))
        features_list.extend(out.detach().cpu().numpy())

        cntr += 1

    return np.array(labels_list), np.array(features_list)


if __name__ == "__main__":

    DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name  = "./Model_Checkpoints/Q2_BestModel-Checkpoint-epoch50.pth"
    split       = 'test'
    inp_size    = 224


    model = ResNet_Repr(model_name).to(DEVICE)
    testLoader = torch.utils.data.DataLoader(VOCDataset(split, inp_size), shuffle = True, batch_size = 100)

    labels_list, features_list = getRepresentations(model, testLoader)

    # Computing 2d Representations
    embedded_features = TSNE().fit_transform(features_list)

    # Transforming the labels list
    transform_arr = np.array([num for num in range(20)])
    labels_list = (np.matmul(labels_list, transform_arr)/np.sum(labels_list, axis = 1)).astype(int)

    # Creating scatter Plot
    fig, ax = plt.subplots(figsize = (15, 10))

    for label in range(20):
        indices = np.where(labels_list == label)
        ax.scatter(embedded_features[indices, 0], embedded_features[indices, 1], label = VOCDataset.CLASS_NAMES[label], alpha=0.5, edgecolors='face')
        ax.legend()

    plt.title("2-D Projection of Class Features")
    fig.savefig("2d_ResNet_Feat_Rep.png")