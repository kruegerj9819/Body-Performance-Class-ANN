import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn import functional as F

bodyPerformance_labels = [
    'A', 'B', 'C', 'D'
]


# https://www.kaggle.com/datasets/kukuroo3/body-performance-data
class BodyData(Dataset):
    def __init__(self):
        # Training set
        df = pd.read_csv("Train_BodyPerformance.csv")
        mean = np.average(df.iloc[:, 0:-1].to_numpy(), axis=0)
        std = np.std(df.iloc[:, 0:-1].to_numpy(), axis=0)
        self.X = torch.tensor((df.iloc[:, 0:-1].to_numpy() - mean) / std, dtype=torch.float)
        self.y = torch.tensor(df.iloc[:, -1].to_numpy(), dtype=torch.long)
        self.len = len(df)

        # Validation set
        df_valid = pd.read_csv("Test_BodyPerformance.csv")
        self.X_valid = torch.tensor((df_valid.iloc[:, 0:-1].to_numpy() - mean) / std, dtype=torch.float)
        self.y_valid = torch.tensor(df_valid.iloc[:, -1].to_numpy(), dtype=torch.long)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len


class BodyNetwork(nn.Module):
    def __init__(self):
        super(BodyNetwork, self).__init__()
        self.in_to_h1 = nn.Linear(12, 32)
        self.h1_to_out = nn.Linear(32, 16)

        self.in_to_h2 = nn.Linear(16, 32)
        self.h2_to_out = nn.Linear(32, 4)

    def forward(self, x):
        x = F.sigmoid(self.in_to_h1(x))
        x = self.h1_to_out(x)

        x = F.relu(self.in_to_h2(x))
        return self.h2_to_out(x)


def trainNN(epochs=10, batch_size=16, lr=0.001, trained_network=None, save_file="bodyHealthNN.pt"):
    # Create dataset
    bd = BodyData()

    # Create data loader
    dl = DataLoader(bd, batch_size=batch_size, shuffle=True, drop_last=True)

    # Create the ANN and load save point from file if it exists
    bodyNN = BodyNetwork()
    if trained_network is not None:
        bodyNN.load_state_dict(torch.load(trained_network))
        bodyNN.train()

    # loss function
    loss_fn = CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(bodyNN.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        for _, data in enumerate(tqdm(dl)):
            X, y = data

            optimizer.zero_grad()

            output = bodyNN(X)

            loss = loss_fn(output, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        with torch.no_grad():
            bodyNN.eval()
            # Only show confusion matrix at the last epoch
            if epoch == epochs - 1:
                y_true = bd.y_valid.numpy()
                y_pred = predictions.numpy()
                cm = confusion_matrix(y_true, y_pred)
                disp = ConfusionMatrixDisplay(cm, display_labels=bodyPerformance_labels)
                disp.plot(cmap='Blues')
                plt.title("Confusion Matrix on Validation Set (Final Epoch)")
                plt.show()
            print(f"Running loss for epoch {epoch + 1} of {epochs}: {running_loss:.4f}")
            predictions = torch.argmax(bodyNN(bd.X), dim=1)
            correct = (predictions == bd.y).sum().item()
            print(f"Accuracy on train set: {correct / len(bd.X):.4f}")
            predictions = torch.argmax(bodyNN(bd.X_valid), dim=1)
            correct = (predictions == bd.y_valid).sum().item()
            print(f"Accuracy on validation set: {correct / len(bd.X_valid):.4f}")
            bodyNN.train()
        running_loss = 0.0


trainNN(epochs=100, lr=0.001)
