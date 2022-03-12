from io import BytesIO

# import lmdb
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision import transforms, utils

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256,rep=1000):

        files = os.listdir(path)
        self.imglist =[]
        for fir in files:
            if fir.endswith('.jpg') or fir.endswith('.png') or fir.endswith('.jpeg'):
                self.imglist.append(os.path.join(path,fir))
        self.imglist = self.imglist*rep
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        img = Image.open(self.imglist[index]).convert('RGB')
        img = self.transform(img)

        return img
