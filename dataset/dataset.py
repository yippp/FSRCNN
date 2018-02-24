import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image
import numpy as np
from math import floor
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class TrainDataset(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(TrainDataset, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

        self.patch = []

        self.dx = 19
        self.dy = 19

        x_cut = 19
        y_cut = 19

        for i in range(len(self.image_filenames)):
            img = load_img(self.image_filenames[i])
            nx = img.size[0] // x_cut
            if nx * x_cut != img.size[0]:
                nx += 1
            ny = img.size[1] // y_cut
            if ny * y_cut != img.size[1]:
                ny += 1
            for x in range(nx):
                for y in range(ny):
                    self.patch.append(img.crop((floor(x/nx*img.size[0]), floor(y/ny*img.size[1]),
                                      floor(x/nx*img.size[0]) + x_cut, floor(y/ny*img.size[1]) + y_cut)))

    def __getitem__(self, index):
        input_image = self.patch[index]
        target = input_image.copy()
        input_image = self.input_transform(input_image)
        target = self.target_transform(target)

        return input_image, target

    def __len__(self):
        return len(self.patch)


class TestDataset(data.Dataset):
    def __init__(self, image_dir):
        super(TestDataset, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.to_tensor = ToTensor()


    def __getitem__(self, index):
        input_image = load_img(self.image_filenames[index])
        x_re = floor((input_image.size[0] - 1) / 3 + 5)
        x = (x_re - 5) * 3 + 1
        if x != input_image.size[0]:
            x = floor(x)
        y_re = floor((input_image.size[1] - 1) / 3 + 5)
        y = (y_re - 5) * 3 + 1
        if y != input_image.size[1]:
            y = floor(y)

        self.crop = CenterCrop((x,y))
        input_image = self.crop(input_image)
        target = input_image.copy()
        target = self.to_tensor(target)

        self.resize = Resize((x_re, y_re))
        input_image = self.resize(input_image)
        input_image = self.to_tensor(input_image)

        return input_image, target

    def __len__(self):
        return len(self.image_filenames)