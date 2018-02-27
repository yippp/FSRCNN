from os.path import join
from .dataset import LoadH5, LoadImg

def get_h5_set(train_set):
    train_dir = join("./dataset", train_set)

    return LoadH5(train_dir)


def get_img_set(test_set):
    test_dir = join("./dataset", test_set)

    return LoadImg(test_dir)
