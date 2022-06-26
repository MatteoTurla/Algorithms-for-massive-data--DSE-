import os
from pathlib import Path
from .dataset import Dataset
from torchvision import transforms as T
import torch

class DataUtils:

    def __init__(self, root_folder, train_len, val_len):
        """
        Data utils class: given the root folder, the number of training image and the number of validation img
        it return a torch dataset ready to use
        """
        self.root_folder = Path(root_folder)
        self.train_len = train_len
        self.val_len = val_len

        faces_path_folder = self.root_folder/'faces'
        comics_path_folder = self.root_folder/'comics'

        faces_path_img = sorted(os.listdir(faces_path_folder))
        comics_path_img = sorted(os.listdir(comics_path_folder))

        self.train_img = [(faces_path_folder/fn, 0) for fn in faces_path_img[:train_len]] + \
                            [(comics_path_folder/fn, 1) for fn in comics_path_img[:train_len]] 

        self.val_img = [(faces_path_folder/fn, 0) for fn in faces_path_img[train_len:train_len+val_len]] + \
                            [(comics_path_folder/fn, 1) for fn in comics_path_img[train_len:train_len+val_len]] 

        self.test_img = [(faces_path_folder/fn, 0) for fn in faces_path_img[train_len+val_len:]] + \
                            [(comics_path_folder/fn, 1) for fn in comics_path_img[train_len+val_len:]] 

    def get_trasformation(self, train=False):
        if train:
            t = T.Compose([
                T.Resize([28]),
                T.RandomHorizontalFlip(),
                T.ConvertImageDtype(torch.float),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]) 
            return t 
        else:
            t  = T.Compose([
                T.Resize([28]),
                T.ConvertImageDtype(torch.float),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            )
            return t 

    def get_training_dataset(self, all=False):
        """
        return the trainig data in a torch Dataset format
        if all=True return train+val data
        """
        t = self.get_trasformation(train=True)
        if all:
            training_dataset = Dataset(self.train_img+self.val_img, transform=t)
        else:
            training_dataset = Dataset(self.train_img, transform=t)

        return training_dataset

    def get_validation_dataset(self):
        t = self.get_trasformation(train=False)
        validation_dataset = Dataset(self.val_img, transform=t)

        return validation_dataset

    def get_testing_dataset(self):
        t = self.get_trasformation(train=False)
        testing_dataset = Dataset(self.test_img, transform=t)

        return testing_dataset      

          
