from os.path import join
from torchvision.transforms import Compose, ToTensor, Resize
from .dataset import TrainDataset, TestDataset
from PIL import Image

def input_transform(input_size):
    return Compose([
        Resize(input_size, interpolation=Image.BICUBIC),
        ToTensor(),
    ])


def target_transform():
    return Compose([
        ToTensor(),
    ])


def get_training_set(train_set):
    train_dir = join("./dataset/train", train_set)

    return TrainDataset(train_dir,
                        input_transform=input_transform(11),
                        target_transform=target_transform())


def get_test_set(test_set):
    test_dir = join("./dataset/test", test_set)

    return TestDataset(test_dir)
