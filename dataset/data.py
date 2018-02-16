from os.path import join
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from .PairRandomCrop import PairRandomCrop
from .dataset import DatasetFromFolder


def input_transform(input_size, target_size):
    return Compose([
        PairRandomCrop(target_size),
        Resize(input_size),
        ToTensor(),
    ])


def target_transform(target_size):
    return Compose([
        PairRandomCrop(target_size),
        ToTensor(),
    ])

def test_input_transform(input_size, target_size):
    return Compose([
        CenterCrop(target_size),
        Resize(input_size),
        ToTensor(),
    ])


def test_target_transform(target_size):
    return Compose([
        CenterCrop(target_size),
        ToTensor(),
    ])


def get_training_set(train_set):
    train_dir = join("./dataset/train", train_set)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(11, 19),
                             target_transform=target_transform(19))


def get_test_set(test_set):
    test_dir = join("./dataset/test", test_set)

    return DatasetFromFolder(test_dir,
                             input_transform=test_input_transform(11, 19),
                             target_transform=test_target_transform(19))
