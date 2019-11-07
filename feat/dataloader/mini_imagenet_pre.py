import os.path as osp
import PIL
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

# use for miniImageNet pre-train
THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/split')

class MiniImageNet(Dataset):

    def __init__(self, setname, args):
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        if args.model_type == 'conv':
            image_size = 84
            if setname == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # Lighting(0.1, imagenet_pca['eigval'], imagenet_pca['eigvec']),
                    transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                         np.array([0.229, 0.224, 0.225])),
                              
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(92),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                         np.array([0.229, 0.224, 0.225]))
                ])            
        elif args.model_type == 'ResNet':
            # for resNet
            image_size = 80
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]
            if setname == 'train':
                self.transform = transforms.Compose([
                    # transforms.Resize(92, interpolation = PIL.Image.BICUBIC),
                    transforms.RandomResizedCrop(image_size),
                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # Lighting(0.1, imagenet_pca['eigval'], imagenet_pca['eigvec']),
                    transforms.Normalize(mean, std)])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(92),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
        elif args.model_type == 'AmdimNet':
            # image augmentation functions
            INTERP = 3
            self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
            rand_crop = \
                transforms.RandomResizedCrop(128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                             interpolation=INTERP)
            col_jitter = transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.25)
            post_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            if setname == 'train':
                self.transform = transforms.Compose([
                    rand_crop,
                    col_jitter,
                    rnd_gray,
                    post_transform
                ])
            else:
                self.transform = transforms.Compose([
                transforms.Resize(146, interpolation=INTERP),
                transforms.CenterCrop(128),
                post_transform
            ])
        else:
            raise Exception("error of model_type")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

