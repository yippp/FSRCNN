from os.path import join
from .dataset import LoadH5, LoadImg

def get_h5_set(train_set):
    """
    Load H5 dataset.
    :param train_set: the filename of the dataset
    :return: the loaded data
    """
    train_dir = join("./dataset", train_set)

    return LoadH5(train_dir)


def get_img_set(test_set):
    """
    Load images file data.
    :param train_set: the folder name of the images in
    :return: the loaded data
    """
    test_dir = join("./dataset", test_set)

    return LoadImg(test_dir)
