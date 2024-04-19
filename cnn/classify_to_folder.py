import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models
from torchvision.datasets import ImageFolder
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision

class HouseFolder(Dataset):
    def __init__(self, csv_file, root_dir, num_class=20, house_count=None, transform=None):
        self.data = pd.read_csv(csv_file) 
        self.root_dir = root_dir
        self.transform = transform
        self.num_class = num_class
        if house_count:
            # random choose house_count
            self.data = self.data[house_count[0]: house_count[1]]
            # reset index
            self.data = self.data.reset_index(drop=True)
        self.bins = pd.qcut(self.data.iloc[:, 2], num_class, labels=False)
        print(self.bins)
        self.images = []
        self.prices = []
        self.addresses = []
        self.image_names = []
        self._load_images()

    def _load_images(self):
        for idx in tqdm.tqdm(range(0, len(self.data))):
            folder_name = self.data.iloc[idx, 0]
            folder_path = os.path.join(self.root_dir, folder_name)
            image_names = os.listdir(folder_path)
            # random choose max 3 images
            image_names = np.random.choice(image_names, min(3, len(image_names)), replace=False)

            for image_name in image_names:
                image_path = os.path.join(folder_path, image_name)
                try :
                    image = ImageFile.Image.open(image_path)
                    self.images.append(image)
                    price = self.bins[idx]
                    self.prices.append(price)
                    self.addresses.append(folder_name)
                    self.image_names.append(image_name)
                except:
                    pass

        # draw label distribution
        plt.hist(self.prices, bins=self.num_class)
        plt.xlabel('Price')
        plt.ylabel('Count')
        plt.savefig('label_distribution.png')
        plt.clf()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        price = self.prices[idx]
        address = self.addresses[idx]
        image_name = self.image_names[idx]
        if self.transform:
            image = self.transform(image)
        return image, price, address, image_name
    
class HouseDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        img, label, address, image_names = self.ds[idx]
        if self.transform:
            img = self.transform(img)  
        return img, label, address, image_names

class RoomTypeCNN(nn.Module):
    def __init__(self, num_classes, transfer_learning=True):
        super(RoomTypeCNN, self).__init__()
        if transfer_learning:
            self.model = models.resnet50(pretrained=True)
            num_ftrs = self.model.fc.in_features
            # for param in self.model.parameters():
            #     param.requires_grad = False
        else:
            self.model = models.resnet50(pretrained=False)
            num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        output = self.model(x)
        return output


if __name__ == '__main__':
    random_seed = 413
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
    ])

    all_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    print('Loading dataset...')
    house_count = [0, 25000]

    dataset = HouseFolder(csv_file='./2021_Residential_cnn.csv', root_dir='./2021/Residential', house_count=house_count)

    print(f'Total number of images: {len(dataset)}')


    dataset = HouseDataset(dataset, transform=all_transform)

    batch_size = 1024

    num_workers = 8

    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)

    model_classifier = RoomTypeCNN(7, transfer_learning=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_classifier.load_state_dict(torch.load('logs/classifier_finetuned.pth'))

    model_classifier = model_classifier.to(device)

    model_classifier.eval()

    with torch.no_grad():
        for imgs, labels, addresses, image_names in tqdm.tqdm(dataloader):
            imgs = imgs.to(device)
            outputs = model_classifier(imgs).argmax(dim=1)
            # save to corresponding folder
            for idx, output in enumerate(outputs):
                address = addresses[idx]
                label = labels[idx]
                folder_path = os.path.join('2021/Residentials', str(output.item()))
                sub_folder_path = os.path.join(folder_path, str(label.item()))
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                if not os.path.exists(sub_folder_path):
                    os.makedirs(sub_folder_path)
                image_path = os.path.join(sub_folder_path, f'{address}_{image_names[idx]}')
                # print(image_path)
                try:
                    torchvision.utils.save_image(imgs[idx], image_path)
                except:
                    pass