from os.path import join
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from .dataset import TrainDataset, TestDataset


def input_transform(input_size, target_size):
    return Compose([
        Resize(input_size),
        ToTensor(),
    ])


def target_transform(target_size):
    return Compose([
        ToTensor(),
    ])


def get_training_set(train_set):
    train_dir = join("./dataset/train", train_set)

    return TrainDataset(train_dir,
                        input_transform=input_transform(11, 19),
                        target_transform=target_transform(19))


def get_test_set(test_set):
    test_dir = join("./dataset/test", test_set)

    return TestDataset(test_dir)
