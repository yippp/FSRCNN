# FSRCNN
Unofficial implementation of FSRCNN using PyTorch.

## Steps
1. Run `data_aug.m` to augment training set.
2. Run `generate_train.m` and `generate_test.m` to generate h5 files for data loader. 
(Note that you may need to change the path in the matlab code for different image source)
3. Run `main.py`.

## Requirement
1. python >= 3.4
2. pytorch >= 0.4
3. torchvision
4. tensorflow

## Reference
Chao Dong, Chen Change Loy, Xiaoou Tang. Accelerating the Super-Resolution Convolutional Neural Network, in Proceedings of European Conference on Computer Vision (ECCV), 2016
