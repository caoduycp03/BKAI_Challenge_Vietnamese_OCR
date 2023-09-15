import os
import torch
import torch.nn as nn
from OCRDataset import OCRDataset
from torchvision.transforms import ToTensor, Resize, Compose, RandomAffine, ColorJitter
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
import torch

def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path='checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(checkpoint, checkpoint_path)

# Usage during training loop:
# After evaluating validation loss
# save_checkpoint(model, optimizer, epoch, val_loss, 'checkpoint.pth')

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-8, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def words_from_labels(labels, char_list):
    """
    converts the list of encoded integer labels to word strings like eg. [12,10,29] returns CAT 
    """
    txt=[]
    for ele in labels:
        if ele == 0: # CTC blank space
            txt.append("")
        else:
            #print(letters[ele])
            txt.append(char_list[ele+1])
    return "".join(txt)

def decode_batch(test_func, word_batch): #take only a sequence once a time
    """
    Takes the Batch of Predictions and decodes the Predictions by Best Path Decoding and Returns the Output
    """
    out = test_func([word_batch])[0] #returns the predicted output matrix of the model
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, :], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = words_from_labels(out_best)
        ret.append(outstr)
    return ret


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    configs = ModelConfigs()
    root = configs.root
    num_epochs = configs.epochs
    batch_size = configs.batch_size
    train_workers = configs.train_workers
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
    
    augment_transform= Compose([RandomAffine(
                                            degrees=(-5, 5),
                                            scale=(0.5, 1.05), 
                                            shear=10),
                                ColorJitter(
                                            brightness=0.5, 
                                            contrast=0.5,
                                            saturation=0.5,
                                            hue=0.5)])

    #split train/val dataset
    dataset = OCRDataset(root = root, max_label_len = max_label_len, train=True, transform=transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=train_workers,
        drop_last=True,
        shuffle=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=train_workers,
        drop_last=True,
        shuffle=True
    )

    if not os.path.isdir(logging):
        shutil.rmtree(logging)
    if not os.path.isdir(trained_models):
        os.mkdir(trained_models)
    writer = SummaryWriter(logging)

    char_list = dataset.char_list
    model = CRNN(num_classes=len(char_list)+1).to(device)
    criterion = nn.CTCLoss(blank=0)
    time_steps = max_label_len #time_steps(seq_len) must >= max_label_len but for simplicity, we use time_steps(seq_len) = max_label_len
    output_lengths = torch.full(size=(batch_size,), fill_value=time_steps, dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    # Early stopping 
    early_stopping = EarlyStopping(patience=20, min_delta=1e-8, restore_best_weights=True)
    
    best_loss = 0
    early_stopping_counter = 0
    early_stopping_patience = 10  # Adjust as needed

    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']  
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0  
    num_iters = len(train_dataloader)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="green")
        for iter, (images, padded_labels, label_lenghts) in enumerate(train_dataloader):
            images = augment_transform(images)
            images = images.to(device)
            padded_labels = padded_labels.to(device)
            #forward
            outputs = model(images)
#Shape:     #output(sequence_length, batch_size, num_classes)
            #padded_labels(batch_size, max_label_len)
            #output_lengths, label_lenghts(batch_size)
            loss_value = criterion(outputs, padded_labels, output_lengths, label_lenghts)
            if torch.isinf(loss_value):
                print(outputs)
                exit()
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss{:3f}".format(epoch+1, num_epochs, iter+1, num_iters, loss_value))
            writer.add_scalar("Train/Loss", loss_value, epoch*num_iters+iter)
            #backward
            optimizer.zero_grad()
            loss_value.backward()  
            optimizer.step()

        model.eval()
        for iter, (images, padded_labels, label_lenghts) in enumerate(val_dataloader):
            images = images.to(device)
            padded_labels = padded_labels.to(device)
            with torch.no_grad():
                predictions = model(images)  
                loss_value = criterion(predictions, padded_labels, output_lengths, label_lenghts)
        writer.add_scalar("Val/Loss", loss_value, epoch)
        # Update learning rate scheduler
        scheduler.step(loss_value)
        
        checkpoint = {
            "epoch": epoch + 1,
            "best_loss" : best_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        # torch.save(checkpoint, "{}/last_crnn.pt".format(trained_models))

        # if loss_value >= best_loss:
        #     torch.save(checkpoint, "{}/best_crnn.pt".format(trained_models))
        #     best_loss = loss_value
        # print('Validate', loss_value)
        if early_stopping(val_loss):
            print("Early stopping triggered.")
        if early_stopping.restore_best_weights:
            checkpoint = torch.load('checkpoint.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        print('Validate', loss_value)

