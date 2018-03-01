from PIL import Image
from dataset.dataset import load_img

img = load_img('./dataset/test/Set5/butterfly_GT.bmp')
# y90 = img.resize((90, 90), resample=Image.BICUBIC)
# y90.save('butterfly90.bmp')
img244 = img.crop((10, 10, 245, 245))
img244.save('butterfly_crop235.bmp')
y86 = img.resize((65, 65), resample=Image.BICUBIC)
y86.save('butterfly65.bmp')
bicubic = y86.resize((256, 256), resample=Image.BICUBIC)
bicubic.save('bicubic65.bmp')
a=0