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

class PriceLevelCNN(nn.Module):
    def __init__(self, num_classes, transfer_learning=True):
        super(PriceLevelCNN, self).__init__()
        if transfer_learning:
            self.model = models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            # for param in self.model.parameters():
            #     param.requires_grad = False
        else:
            self.model = models.resnet18(pretrained=False)
            num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        output = self.model(x)
        return output

    
# class RoomTypeCNN(nn.Module):
#     def __init__(self, num_classes, transfer_learning=True):
#         super(RoomTypeCNN, self).__init__()
#         if transfer_learning:
#             self.model = models.resnet50(pretrained=True)
#             num_ftrs = self.model.fc.in_features
#             # for param in self.model.parameters():
#             #     param.requires_grad = False
#         else:
#             self.model = models.resnet152(pretrained=False)
#             num_ftrs = self.model.fc.in_features

#         self.model.fc = nn.Sequential(
#             nn.Linear(num_ftrs, num_classes),
#             nn.LogSoftmax(dim=1)
#         )

#     def forward(self, x):
#         output = self.model(x)
#         return output

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

    roomtype = 6

    dataset = ImageFolder(f'./2021/Residentials/{roomtype}', transform=None)

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

    model_pricelevel = PriceLevelCNN(num_classes=20, transfer_learning=False)

    lr = 0.001
    num_epochs = 200
    save_epoch = 4
    weight_decay = 0.0001

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model_pricelevel.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=num_epochs, steps_per_epoch=len(train_loader))
    # scheduler = optim.lr_scheduler.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_pricelevel = model_pricelevel.to(device)

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
        
        model_pricelevel.train()

        running_loss = 0.0

        train_batch_accuracy = []

        for imgs, labels in tqdm.tqdm(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model_pricelevel(imgs)
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

        model_pricelevel.eval()
        val_loss = 0.0
        print_first = False
        with torch.no_grad():
            val_batch_accuracy = []
            for imgs, labels in tqdm.tqdm(val_loader):
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model_pricelevel(imgs)
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
        if (epoch+1) % save_epoch == 0:
            torch.save(model_pricelevel, f'{log_dir}/type_{roomtype}_pricelevel_finetuned_{epoch+1}.pth')

    plt.plot(training_loss, label='Training loss')
    plt.plot(validation_loss, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{log_dir}/loss-type_{roomtype}_pricelevel.png')

    # clear the current figure
    plt.clf()

    plt.plot(training_accuracy, label='Training accuracy')
    plt.plot(validation_accuracy, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{log_dir}/accuracy-type_{roomtype}_pricelevel.png')