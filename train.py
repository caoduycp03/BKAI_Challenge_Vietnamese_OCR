import os
import torch
import torch.nn as nn
from OCRDataset import OCRDataset
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from model import CRNN
import itertools
import numpy as np
from argparse import ArgumentParser
from config import ModelConfigs
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil
import warnings
warnings.simplefilter("ignore")

# Define your DATA LOADERS for training and testing
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

configs = ModelConfigs()
root = configs.root
num_epochs = configs.epochs
batch_size = configs.batch_size
max_label_len = configs.max_label_len
height = configs.height
width = configs.width
learning_rate = configs.learning_rate
logging = configs.logging
trained_models = configs.trained_models
checkpoint = configs.checkpoint

transform = Compose([
    Resize((height,width)),
    ToTensor(),
     ])

#split train/val dataset
dataset = OCRDataset(root = root, train=True, transform=transform)  # Replace with your dataset
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=4,
    drop_last=True,
    shuffle=True
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=4,
    drop_last=True,
    shuffle=True
)

# Define your model
model = CRNN(time_steps=max_label_len, num_classes=len(dataset.char_list)+1).to(device)

# Define your loss function and optimizer
criterion = nn.CTCLoss(blank=0)
output_lengths = torch.full(size=(batch_size,), fill_value=max_label_len, dtype=torch.long)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
best_loss = 0



# Define your callbacks
####################
reduce_learning = ReduceLROnPlateau(optimizer = optimizer, 
                                    mode='max', 
                                    factor=0.5, 
                                    patience=2, 
                                    verbose=True)
early_stop = EarlyStopping(patience=11, 
                           verbose=True)
callbacks = {
    'reduce_learning': reduce_learning,
    'early_stop': early_stop
}
####################
# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        inputs, labels,_ = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update the running loss
        running_loss += loss.item()

    # Print the average loss for this epoch
    print(f"Epoch {epoch+1} - Average Loss: {running_loss / len(train_dataloader)}")

    # Perform validation after each epoch
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_dataloader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print the validation accuracy
        print(f"Epoch {epoch+1} - Validation Accuracy: {100 * correct / total}%")

# Save the trained model
torch.save(model.state_dict(), "path/to/save/model.pth")
