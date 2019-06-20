import torch.utils.data as data
from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop, Resize
import h5py
import numpy as np
from torch import from_numpy
from math import floor
from os import listdir
from os.path import join

def is_image_file(filename):
    """
    Check wheather the file is a image file.
    :param filename: name of the file
    :return: bool value shows that whether it is a image
    """
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    """
    Load the image and get the luminance data.
    :param filepath: path of the image.
    :return: luminance data
    """
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class LoadH5(data.Dataset):
    def __init__(self, image_h5):
        super(LoadH5, self).__init__()

        self.to_tensor = ToTensor()

        self.input_patch = []
        self.target_patch = []

        with h5py.File(image_h5, 'r') as hf:
            self.input_patch = np.array(hf.get('data'))
            self.target_patch = np.array(hf.get('label'))

    def __getitem__(self, index):
        input_image = self.input_patch[index]
        target_image = self.target_patch[index]
        return from_numpy(input_image), from_numpy(target_image)

    def __len__(self):
        return len(self.input_patch)

class LoadImg(data.Dataset):
    def __init__(self, image_dir):
        super(LoadImg, self).__init__()
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