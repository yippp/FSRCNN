from PIL import Image
from dataset.dataset import load_img

img = load_img('./dataset/test/Set5/butterfly_GT.bmp')
# y90 = img.resize((90, 90), resample=Image.BICUBIC)
# y90.save('butterfly90.bmp')
img244 = img.crop((2, 2, 253, 253))
img244.save('butterfly_crop251.bmp')
y86 = img.resize((127, 127), resample=Image.BICUBIC)
y86.save('butterfly127.bmp')
bicubic = y86.resize((256, 256), resample=Image.BICUBIC)
bicubic.save('bicubic127.bmp')
a=0