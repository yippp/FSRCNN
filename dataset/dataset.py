import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

        self.patch = []

        self.dx = 19
        self.dy = 19

        for i in range(len(self.image_filenames)):
            x = 0
            y = 0
            img = load_img(self.image_filenames[i])
            while x <= img.size[0] - self.dx:
                while y <= img.size[1] - self.dy:
                    patch = img.crop((y, x, y + self.dy, x + self.dx))
                    check = np.asarray(patch)
                    if not check.max() == 0:
                        self.patch.append(patch)
                    y += self.dy
                x += self.dx
                y = 0

    def __getitem__(self, index):
        input_image = self.patch[index]
        target = input_image.copy()
        input_image = self.input_transform(input_image)
        target = self.target_transform(target)

        return input_image, target

    def __len__(self):
        return len(self.patch)
