from PIL import Image
from dataset.dataset import load_img

img = load_img('./dataset/test/Set5/butterfly_GT.bmp')
y90 = img.resize((90, 90), resample=Image.BICUBIC)
y90.save('butterfly90.bmp')
bicubic = y90.resize((256, 256), resample=Image.BICUBIC)
bicubic.save('bicubic.bmp')
a=0