import torch
from torchvision import datasets
from typing import Tuple
import os
from os.path import isfile, join
from PIL import Image
from torchvision import transforms, utils

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths"""

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class SingleImageFolder(torch.utils.data.Dataset):
    """Custom dataset of all images in single folder"""

    def __init__(self, folder_path: str, transform = None):
        self.transform = transform or transforms.ToTensor()
        self.file_names = [join(folder_path, f) for f in os.listdir(folder_path)]
        self.file_names.sort()
        
    def __getitem__(self, index) -> Tuple[torch.Tensor, str]:
        return (self._pil_loader(self.file_names[index]), self.file_names[index])

    def __len__(self):
        return len(self.file_names)

    def _pil_loader(self, path) -> torch.Tensor:
        if path is None: return None
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            return self.transform(img)