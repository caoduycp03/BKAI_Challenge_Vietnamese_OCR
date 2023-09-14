from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torchvision.transforms import ToTensor, Resize, Compose
import torchvision.transforms.functional as F
import os
import numpy as np
from PIL import Image


def encode_to_num(text, char_list):
    encoded_label = []
    for char in text:
        encoded_label.append(char_list.index(char)+1)
    return encoded_label

class OCRDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.train = train
        self.transform = transform
        if train:
            dir = os.path.join(root, 'new_train')
            paths = os.listdir(dir)
            image_files = [os.path.join(dir, path) for path in paths]
            label_file = 'data\\train_gt.txt'
        else:
            dir = os.path.join(root, 'new_public_test')
            paths = os.listdir(dir)
            image_files = [os.path.join(dir, path) for path in paths]
        
        self.images_path = image_files
        if train:
            self.labels = []
            with open(label_file, encoding='utf-8') as f:
                self.labels = [line.split()[1] for line in f.readlines()]
            char_list= set()
            for label in self.labels:
                char_list.update(set(label))
            self.char_list = sorted(char_list)
            for i in range(len(self.labels)):
                self.labels[i] = encode_to_num(self.labels[i], self.char_list)

    def __len__(self):
        return len(self.images_path)
    def __getitem__(self, idx):      
        image_path = self.images_path[idx]
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        if self.train:
            label = self.labels[idx]
            max_seq_len = 32
            padded_label = np.squeeze(pad_sequences([label], maxlen=max_seq_len, padding='post', value = 0))
            return image, padded_label, len(label)
        else:
            return image
        

if __name__ == '__main__':
    transform = Compose([
        Resize((64,128)),
        ToTensor(),
        ])

    train_dataloader = DataLoader(
        dataset=OCRDataset(root = "data", train=True, transform=transform),
        batch_size=8,
        num_workers=4,
        drop_last=True,
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=OCRDataset(root = "data", train=False, transform=transform),
        batch_size=8,
        num_workers=4,
        drop_last=True,
        shuffle=True
    )
    
    ocr = OCRDataset(root = "data", train=True, transform=transform)
    # image, label, length = ocr.__getitem__(1)
    # print(image.shape)
    # print(label)
    max_len = 0
    for i in ocr.labels:
        if len(i) > max_len:
            max_len = len(i)
    print(max_len)
    

    # height = []
    # width = []
    # for i in range(103000):
    #     image, label= ocr.__getitem__(i)
    #     arr = np.asarray(image)
    #     width.append(arr.shape[1])
    #     height.append(arr.shape[0]) 
    # print(np.mean(height)) #71.8999708737864 => 64 
    # print(np.mean(width)) #131.1066213592233 => 128


    # for images, labels in train_dataloader:
    #     print(images.shape)
    #     print(labels.shape)
