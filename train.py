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

    if not os.path.isdir(logging):
        shutil.rmtree(logging)
    if not os.path.isdir(trained_models):
        os.mkdir(trained_models)
    writer = SummaryWriter(logging)

    char_list = dataset.char_list
    model = CRNN(time_steps=max_label_len, num_classes=len(char_list)+1).to(device)
    criterion = nn.CTCLoss(blank=0)
    output_lengths = torch.full(size=(batch_size,), fill_value=max_label_len, dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = 0
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
            images = images.to(device)
            padded_labels = padded_labels.to(device)
            #forward
            outputs = model(images)
#Shape:     #output(sequence_length, batch_size, num_classes)
            #padded_labels(batch_size, max_label_len)
            #output_lengths, label_lenghts(batch_size)
            loss_value = criterion(outputs, padded_labels, output_lengths, label_lenghts)
            if torch.isinf(loss_value) or torch.isnan(loss_value):
                print(outputs, padded_labels)
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss{:3f}".format(epoch+1, num_epochs, iter+1, num_iters, loss_value))
            writer.add_scalar("Train/Loss", loss_value, epoch*num_iters+iter)
            #backward
            optimizer.zero_grad()
            loss_value.backward()  
            optimizer.step()

        model.eval()
        for iter, (images, padded_labels) in enumerate(val_dataloader):
            images = images.to(device)
            padded_labels = padded_labels.to(device)
            with torch.no_grad():
                predictions = model(images)  
                loss_value = criterion(predictions, padded_labels, output_lengths, label_lenghts)
        writer.add_scalar("Val/Loss", loss_value, epoch)
        checkpoint = {
            "epoch": epoch + 1,
            "best_loss" : best_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_crnn.pt".format(trained_models))

        if loss_value >= best_loss:
            torch.save(checkpoint, "{}/best_crnn.pt".format(trained_models))
            best_loss = loss_value
        print('Validate', loss_value)


