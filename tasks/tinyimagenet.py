from __future__ import print_function

import os
import pickle
import zipfile
import urllib.request
from PIL import Image
import torch.nn.functional as F

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from dataloader import iidLoader, dirichletLoader, byLabelLoader

try:
    from tqdm import tqdm
except ImportError:
    print('tqdm not found. Please install it for a download progress bar: pip install tqdm')
    tqdm = None

# Normalization constants for Tiny ImageNet
TINY_IMAGENET_MEAN = [0.4802, 0.4481, 0.3975]
TINY_IMAGENET_STD = [0.2302, 0.2265, 0.2262]

def _download_and_extract_archive(url, download_root, extract_root, filename):
    """
    Downloads and extracts an archive file.
    Args:
        url (str): URL to download.
        download_root (str): Path to download the archive.
        extract_root (str): Path to extract the archive.
        filename (str): Name of the archive file.
    """
    # Ensure the download root directory exists
    os.makedirs(download_root, exist_ok=True)
    
    # If the extracted folder already exists, do nothing
    if os.path.exists(extract_root):
        print(f"Dataset already found and extracted at '{extract_root}'.")
        return

    archive_path = os.path.join(download_root, filename)

    # Download the file if it doesn't exist
    if not os.path.exists(archive_path):
        print(f"Downloading {url} to {archive_path}")
        
        # A TQDM hook for urlretrieve
        class TqdmUpTo(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        # Use tqdm if available
        if tqdm:
            with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
                urllib.request.urlretrieve(url, filename=archive_path, reporthook=t.update_to)
        else:
            urllib.request.urlretrieve(url, filename=archive_path)

    # Extract the file
    print(f"Extracting {archive_path} to {download_root}")
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(path=download_root)
    print("Extraction complete.")


class TinyImageNet(torch.utils.data.Dataset):
    """
    Custom Dataset for Tiny ImageNet
    """
    def __init__(self, root, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.data_dir = os.path.join(self.root, 'train' if self.train else 'val')
        
        # Load class names and create mapping
        self.wnids_path = os.path.join(self.root, 'wnids.txt')
        self.words_path = os.path.join(self.root, 'words.txt')
        self.class_to_idx, self.idx_to_class, self.classes = self._create_class_mapping()

        self.samples = self._make_dataset()

    def _create_class_mapping(self):
        with open(self.wnids_path, 'r') as f:
            wnids = [x.strip() for x in f.readlines()]
        
        class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}
        idx_to_class = {i: wnid for i, wnid in enumerate(wnids)}

        with open(self.words_path, 'r') as f:
            lines = f.readlines()
            wnid_to_name = {}
            for line in lines:
                wnid, name = line.strip().split('\t', 1)
                wnid_to_name[wnid] = name
        
        classes = [wnid_to_name[idx_to_class[i]] for i in range(len(wnids))]
        
        return class_to_idx, idx_to_class, classes

    def _make_dataset(self):
        samples = []
        if self.train:
            train_wnids = os.listdir(self.data_dir)
            for wnid in train_wnids:
                class_idx = self.class_to_idx[wnid]
                class_dir = os.path.join(self.data_dir, wnid, 'images')
                for fname in os.listdir(class_dir):
                    path = os.path.join(class_dir, fname)
                    samples.append((path, class_idx))
        else: # Validation set
            val_ann_path = os.path.join(self.root, 'val', 'val_annotations.txt')
            with open(val_ann_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    img_name, wnid = parts[0], parts[1]
                    path = os.path.join(self.data_dir, 'images', img_name)
                    class_idx = self.class_to_idx[wnid]
                    samples.append((path, class_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, target

class Net(nn.Module):
    """
    ResNet18 for Tiny ImageNet
    """
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 200)

    def forward(self, x):
        return self.model(x)
    
def getDataset(train):
    """
    Handles downloading, extracting, and creating the TinyImageNet dataset object.
    """
    # --- NEW AUTOMATION LOGIC ---
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    download_root = './data'
    filename = 'tiny-imagenet-200.zip'
    extract_root = os.path.join(download_root, 'tiny-imagenet-200')
    
    _download_and_extract_archive(url, download_root, extract_root, filename)
    # --- END OF NEW LOGIC ---

    if train:
        transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
        ])
        
    # The TinyImageNet class now points to the automatically extracted folder
    dataset = TinyImageNet(root=extract_root, train=train, transform=transform)
    return dataset


def basic_loader(num_clients, loader_type):
    dataset = getDataset(train=True)
    dataset.targets = [s[1] for s in dataset.samples]
    return loader_type(num_clients, dataset)


def train_dataloader(num_clients, loader_type='iid', store=True, path='./data/loader.pk'):
    loader_path = path.replace('.pk', '_tiny-imagenet.pk')
    
    if loader_type == 'iid':
        loader_class = iidLoader
    elif loader_type == 'byLabel':
        loader_class = byLabelLoader
    elif loader_type == 'dirichlet':
        loader_class = dirichletLoader
    else:
        raise ValueError("Unknown loader type")

    if store and os.path.exists(loader_path):
        with open(loader_path, 'rb') as handle:
            loader = pickle.load(handle)
    else:
        print('Initialize a data loader for Tiny ImageNet')
        loader = basic_loader(num_clients, loader_class)
        if store:
            print(f'Save the dataloader {loader_path}')
            with open(loader_path, 'wb') as handle:
                pickle.dump(loader, handle)
    return loader


def test_dataloader(test_batch_size):
    test_dataset = getDataset(train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=True
    )
    return test_loader