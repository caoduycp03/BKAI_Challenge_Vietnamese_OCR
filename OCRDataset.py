from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torchvision.transforms import ToTensor, Resize, Compose
import torchvision.transforms.functional as F
import os
import numpy as np
from PIL import Image


train_folder_path = '/kaggle/input/handwritten-ocr/training_data/new_train' 
test_folder_path = '/kaggle/input/handwritten-ocr/public_test_data/new_public_test'
label_file_path = '/kaggle/input/handwriting/train_gt.txt'
root = '/kaggle/input/handwritten-ocr'


def encode_to_num(text, char_list):
    encoded_label = []
    for char in text:
        encoded_label.append(char_list.index(char))
    return encoded_label

class OCRDataset(Dataset):
    def __init__(self, root, max_label_len, train=True, transform=None):
        self.max_label_len = max_label_len

        self.train = train
        self.transform = transform
        if train:
            dir = os.path.join(root, train_folder_path)
            paths = os.listdir(dir)
            image_files = [os.path.join(dir, path) for path in paths]
            label_file = label_file_path
        else:
            dir = os.path.join(root, test_folder_path)
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
            padded_label = np.squeeze(pad_sequences([label], maxlen=self.max_label_len, padding='post', value = 0))
            return image, padded_label, len(label)
        else:
            return image
        


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
    

for images, labels, nothing in train_dataloader:
    print(images)
    print(labels)
    print(nothing)
    break

for images, labels, nothing in train_dataloader:
        print(images.shape)
        break