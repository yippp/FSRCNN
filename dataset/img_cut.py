from PIL import Image
import os
import numpy as np
from math import floor

x_cut = 19
y_cut = 19

imgTypes = [".bmp"]

for root, dirs, files in os.walk("./test/Set5"):
    for afile in files:
        ffile = root + "/" + afile
        print('cutting', afile)

        if ffile[ffile.rindex("."):].lower() in imgTypes:
            im = Image.open(ffile)

            nx = im.size[0] // x_cut
            if nx * x_cut != im.size[0]:
                nx += 1
            ny = im.size[1] // y_cut
            if ny * y_cut != im.size[1]:
                ny += 1

            n = 1

            for x in range(nx):
                for y in range(ny):
                    name = "./test/5cut/" + afile.replace('.bmp', '_' + str(n) + '.bmp')
                    im2 = im.crop((floor(x / nx * im.size[0]), floor(y / ny * im.size[1]),
                                   floor(x / nx * im.size[0]) + x_cut, floor(y / ny * im.size[1]) + y_cut))
                    check = np.asarray(im2)
                    if not check.max() == 0:
                        im2.save(name)
                    n += 1
