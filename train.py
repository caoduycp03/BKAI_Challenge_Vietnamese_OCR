import torch
import torch.nn as nn
from OCRDataset import OCRDataset
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from model import CRNN

if __name__ == '__main__':
    num_epochs = 100
    batch_size = 8
    max_label_len = 16

    transform = Compose([
        Resize((64,128)),
        ToTensor(),
        ])
    
    #split train/val dataset
    dataset = OCRDataset(root = "data", train=True, transform=transform)  # Replace with your dataset
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
    char_list = dataset.char_list
    model = CRNN(num_classes=len(char_list)+1)
    criterion = nn.CTCLoss(blank=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_iters = len(train_dataloader)
    if torch.cuda.is_available():
        model.cuda()
    for epoch in range(num_epochs):
        model.train()
        for iter, (images, padded_labels, label_lenghts) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            outputs = outputs.permute(1,0,2)
            #output(sequence_length, batch_size, num_classes)
            #padded_labels(batch_size, max_label_len)
            #output_lengths, label_lenghts(batch_size)
            output_lengths = torch.full(size=(batch_size,), fill_value=max_label_len, dtype=torch.long)
            loss_value = criterion(outputs, padded_labels, output_lengths, label_lenghts)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            if iter+1%10:
                print("Epoch {}/{}. Iteration {}/{}. Loss{}".format(epoch+1,num_epochs, iter+1, num_iters, loss_value))
                
        # model.eval()
        # all_predictions=[]
        # all_labels=[]





