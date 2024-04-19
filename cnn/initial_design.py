import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.tensorboard import SummaryWriter

# class HouseFolder(Dataset):
#     def __init__(self, csv_file, root_dir, size=None, transform=None):
#         self.data = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#         if size:
#             self.data = self.data[:size]

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         folder_name = self.data.iloc[idx, 0]
#         folder_path = os.path.join(self.root_dir, folder_name)
        
#         image_names = os.listdir(folder_path)

#         image = Image.open(os.path.join(folder_path, image_names[np.random.randint(0, len(image_names))]))

#         if self.transform:
#             image = self.transform(image)

#         price = int(self.data.iloc[idx, 2] / 100000)

#         return image, price

class HouseFolder(Dataset):
    def __init__(self, csv_file, root_dir, num_class=20, house_count=None, transform=None):
        self.data = pd.read_csv(csv_file) 
        self.root_dir = root_dir
        self.transform = transform
        self.num_class = num_class
        if house_count:
            # random choose house_count
            self.data = self.data.sample(n=house_count)
            # reset index
            self.data = self.data.reset_index(drop=True)
        self.bins = pd.qcut(self.data.iloc[:, 2], num_class, labels=False)
        self.images = []
        self.prices = []
        self._load_images()

    def _load_images(self):
        for idx in range(len(self.data)):
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
        if self.transform:
            image = self.transform(image)
        return image, price
    
class HouseDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)  
        return img, label

class CNN(nn.Module):
    def __init__(self, num_classes, transfer_learning=True):
        super(CNN, self).__init__()
        if transfer_learning:
            self.model = models.resnet152(pretrained=True)
            num_ftrs = self.model.fc.in_features
            # for param in self.model.parameters():
            #     param.requires_grad = False
        else:
            self.model = models.resnet152(pretrained=False)
            num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes * 10),
            # nn.Dropout(0.5),
            nn.Linear(num_classes * 10, num_classes),
            # nn.Linear(num_ftrs, num_classes),
            # nn.Dropout(0.5),
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

    house_count = None

    dataset = HouseFolder(csv_file='./2021_Residential_cnn.csv', root_dir='./2021/Residential', house_count=house_count)

    print(f'Total number of images: {len(dataset)}')

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataset = HouseDataset(train_dataset, transform=train_transform)
    val_dataset = HouseDataset(val_dataset, transform=all_transform)

    batch_size = 64

    num_workers = 4

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size*2, shuffle=True, num_workers=num_workers)

    model = CNN(num_classes=25)

    lr = 0.1
    num_epochs = 200
    weight_decay = 0.0001

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=num_epochs, steps_per_epoch=len(train_loader))
    # scheduler = optim.lr_scheduler.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []

    # write tensorboard to folder
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f'./logs/{timestamp}'
    writer = SummaryWriter(log_dir)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        train_batch_accuracy = []

        for imgs, labels in tqdm.tqdm(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            # scheduler.step()

            running_loss += loss.item() * imgs.size(0)
            train_batch_accuracy.append((outputs.argmax(dim=1) == labels).float().mean().item())


        # if scheduler:
        #     scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Accuracy/train', np.mean(train_batch_accuracy), epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}, Training Accuracy: {np.mean(train_batch_accuracy):.4f}")

        training_accuracy.append(np.mean(train_batch_accuracy))

        # Save the loss for plotting
        training_loss.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        print_first = False
        with torch.no_grad():
            val_batch_accuracy = []
            for imgs, labels in tqdm.tqdm(val_loader):
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                if print_first:
                    print_first = False
                    print(outputs.argmax(dim=1))
                    print(labels)
                    print(outputs.argmax(dim=1) == labels)
                val_loss += loss.item() * imgs.size(0)
                val_batch_accuracy.append((outputs.argmax(dim=1) == labels).float().mean().item())

            val_loss /= len(val_loader.dataset)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', np.mean(val_batch_accuracy), epoch)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {np.mean(val_batch_accuracy):.4f}")

        validation_accuracy.append(np.mean(val_batch_accuracy))

        # Save the loss for plotting
        validation_loss.append(val_loss)

        # Save the model for each 25th epoch
        if (epoch+1) % 25 == 0:
            torch.save(model.state_dict(), log_dir + f'/model_{epoch+1}.pth')

    plt.plot(training_loss, label='Training loss')
    plt.plot(validation_loss, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss-step.png')

    # clear the current figure
    plt.clf()

    plt.plot(training_accuracy, label='Training accuracy')
    plt.plot(validation_accuracy, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy-step.png')