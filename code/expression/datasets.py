
import torch.utils.data as data
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random


class RAFDBDataset(torch.utils.data.Dataset):
    def __init__(self, choose, data_path, label_path, train_sample_num=None, transform=None, img_size=112):
        self.image_paths = []
        self.labels = []

        self.data_path = data_path
        self.label_path = label_path

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([img_size, img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.train = True if choose == "train" else False

        if choose == "train":
            self.sample_num = train_sample_num

            with open(self.label_path, "r") as f:
                data = f.readlines()

            flag = True
            while flag:
                for i in range(0, len(data)):
                    line = data[i].strip('\n').split(" ")

                    image_name = line[0]
                    sample_temp = image_name.split("_")[0]

                    if self.train and sample_temp == "train":
                        ### added
#                         image_name = image_name.split(".")[0]
#                         image_name += '_aligned.jpg'
#                         image_name = os.path.join('Image/aligned', image_name)
                        
                        image_path = os.path.join(self.data_path, image_name)
                        self.image_paths.append(image_path)
                        self.labels.append(int(line[1]) - 1)

                    if len(self.labels) == self.sample_num:
                        flag = False
                        break

        elif choose == "test":
            with open(self.label_path, "r") as f:
                data = f.readlines()

            for i in range(0, len(data)):
                line = data[i].strip('\n').split(" ")

                image_name = line[0]
                sample_temp = image_name.split("_")[0]

                if not self.train and sample_temp == "test":
                    ### added
#                     image_name = image_name.split(".")[0]
#                     image_name += '_aligned.jpg'
#                     image_name = os.path.join('Image/aligned', image_name)
                    
                        
                    image_path = os.path.join(self.data_path, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(int(line[1]) - 1)

        self.labels = np.asarray(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = torch.tensor(self.labels[idx])
        img1 = transforms.RandomHorizontalFlip(p=1.0)(img)
        return img, label, img1

    def __len__(self):
        return len(self.image_paths)
    
    
    
    
    
    
class FERPlusDataset(torch.utils.data.Dataset):
    def __init__(self, choose, data_path, label_path, train_sample_num=None, transform=None, img_size=112):
        self.image_paths = []
        self.labels = []

        self.data_path = data_path
        self.label_path = label_path

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([img_size, img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.train = True if choose == "train" else False

        if choose == "train":
            self.sample_num = train_sample_num

            with open(self.label_path, "r") as f:
                data = f.readlines()

            flag = True
            while flag:
                for i in range(0, len(data)):
                    line = data[i].strip('\n').split(" ")

                    image_name = line[0]
                    if self.train:                        
                        image_path = os.path.join(self.data_path, image_name)
                        self.image_paths.append(image_path)
                        self.labels.append(int(line[1]))

                    if len(self.labels) == self.sample_num:
                        flag = False
                        break

        elif choose == "test":
            with open(self.label_path, "r") as f:
                data = f.readlines()

            for i in range(0, len(data)):
                line = data[i].strip('\n').split(" ")

                image_name = line[0]
                if not self.train:                    
                    image_path = os.path.join(self.data_path, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(int(line[1]))

        self.labels = np.asarray(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = torch.tensor(self.labels[idx])
        img1 = transforms.RandomHorizontalFlip(p=1.0)(img)
        return img, label, img1

    def __len__(self):
        return len(self.image_paths)
    
    
    
    


class AffectNetDataset(torch.utils.data.Dataset):
    def __init__(self, choose, data_path, label_path, train_sample_num=None, transform=None, img_size=112):
        self.image_paths = []
        self.labels = []

        self.data_path = data_path
        self.label_path = label_path


        self.transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.RandomErasing(scale=(0.02, 0.1))
        ])
        

        if choose == "train":
            self.sample_num = train_sample_num

            with open(self.label_path, "r") as f:
                data = f.readlines()

            flag = True
            while flag:

                for i in range(len(data)):
                    line = data[i].split()

                    image_path = os.path.join(self.data_path, line[0])
                    self.image_paths.append(image_path)
                    self.labels.append(int(line[1]))

                    if len(self.labels) == self.sample_num:
                        flag = False
                        break

        elif choose == "test":
            with open(self.label_path, "r") as f:
                data = f.readlines()

            for i in range(len(data)):
                line = data[i].split()

                image_path = os.path.join(self.data_path, line[0])
                self.image_paths.append(image_path)
                self.labels.append(int(line[1]))


    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = torch.tensor(self.labels[idx])
        img1 = transforms.RandomHorizontalFlip(p=1.0)(img)
        return img, label, img1

    def __len__(self):
        return len(self.image_paths)
    
    def get_labels(self):
        label = self.labels
        return label