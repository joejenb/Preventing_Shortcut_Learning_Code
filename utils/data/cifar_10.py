from torch.utils.data import Dataset
import os
import os.path
import torch
import numpy as np


class Cifar10_2_Dataset(Dataset):

    def __init__(self, data_dir="datasets/cifar-10-2", train=False):
        #Assumes dataset has been downloaded and organised appropriately
        train_loc = os.path.join(data_dir, "train.npz")
        test_loc = os.path.join(data_dir, "test.npz")
        data_file = train_loc if train else test_loc
        data = np.load(data_file)

        self.images = data["images"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = torch.Tensor(np.asarray(self.images[idx]).astype('float')).permute(2, 0, 1)
        label = torch.Tensor(np.asarray(self.labels[idx]).astype('float'))
        return (image, label)
    


class Cifar_10_C:

    def __init__(self, cifar10_data_dir="datasets/", data_dir="datasets/cifar-10-c"):
        #Assumes dataset has been downloaded and organised appropriately
        self.data_dir = data_dir
        self.cifar10_data_dir = cifar10_data_dir
        self.corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
            'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
        ]

    def get_dataset_names(self):
        return self.corruptions
    
    def get_sub_dataset(self, test_data, sub_dataset_name):
        full_data_pth = os.path.join(self.data_dir, f"{sub_dataset_name}.npy")
        full_labels_pth = os.path.join(self.data_dir, "labels.npy")

        test_data.data = np.load(full_data_pth)
        test_data.targets = torch.LongTensor(np.load(full_labels_pth))

        return test_data
