"""
Crop the butterfly image for prediction during training.
"""

from PIL import Image
from dataset.dataset import load_img

img = load_img('./dataset/test/Set5/butterfly_GT.bmp')
# y90 = img.resize((90, 90), resample=Image.BICUBIC)
# y90.save('butterfly90.bmp')
img244 = img.crop((6, 6, 250, 250))
img244.save('butterfly_crop244.bmp')
y86 = img.resize((120, 120), resample=Image.BICUBIC)
y86.save('butterfly120.bmp')
bicubic = y86.resize((256, 256), resample=Image.BICUBIC)
bicubic.save('bicubic120.bmp')
