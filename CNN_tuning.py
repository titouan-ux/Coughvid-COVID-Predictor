import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import shutil
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import sklearn.metrics as metrics
import gc
from sklearn.model_selection import RandomizedSearchCV
from random import sample
import random
from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from Models.CNN import ConvNet

random.seed(10)
torch.manual_seed(10)

# Set image and metadata paths
path_img_dir_train = "Data/MelSpecs_train"
df_label_train = pd.read_csv("Data/MelSpecs_labels_train.csv")


class MelSpectrogramDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx]["label"]
        filename = self.df.iloc[idx]["filename"]

        # Read image
        img_path = os.path.join(self.img_dir, filename)
        img = cv2.imread(img_path)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # Convert (H, W, C) to (C, H, W)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Get one image to determine input dimensions
sample_img, _ = train_dataset[0]  # First image and label
input_dim = sample_img.shape  # (C, H, W)
print("Input shape:", input_dim)


param_grid = {"lr": [0.001, 0.0001], "batch_size": [256], "dropout": [0.5]}

param_combinations = list(
    product(param_grid["lr"], param_grid["batch_size"], param_grid["dropout"])
)

# Randomly sample k combinations
random_sample = sample(param_combinations, 10)
print(random_sample)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=10)
results = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for idx, (lr, batch_size, dropout) in enumerate(random_sample):
    print(f"\n=== Combination {idx+1}: LR={lr}, BS={batch_size}, Dropout={dropout} ===")
    val_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_label_train, df_label_train["label"])
    ):
        train_fold = df_label_train.iloc[train_idx].reset_index(drop=True)
        val_fold = df_label_train.iloc[val_idx].reset_index(drop=True)

        train_dataset = MelSpectrogramDataset(train_fold, path_img_dir_train)
        val_dataset = MelSpectrogramDataset(val_fold, path_img_dir_train)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        model = ConvNet(input_dim=input_dim, drop_out=dropout).to(device)
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(50):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            y_true, y_pred = [], []
            val_loss_total = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch).squeeze()
                    val_loss_total += criterion(outputs, y_batch).item()
                    preds = torch.sigmoid(outputs).cpu().numpy()
                    y_pred.extend((preds >= 0.5).astype(int))
                    y_true.extend(y_batch.cpu().numpy())

            avg_val_loss = val_loss_total / len(val_loader)

        score = accuracy_score(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred)
        val_scores.append(score)
        print(f"Fold {fold} done. Accuracy Score: {score:.4f}, AUC: {auc_score:.4f}")

    avg_score = np.mean(val_scores)
    results.append(((lr, batch_size, dropout), avg_score))
    print(
        f"Params: LR={lr}, BS={batch_size}, Dropout={dropout} -> Avg Accuracy: {avg_score:.4f}"
    )
# Save results to CSV
results_df = pd.DataFrame(results, columns=["params", "avg_accuracy"])
results_df.to_csv("tuning_results.csv", index=False)
