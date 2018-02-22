from PIL import Image
import os
import numpy as np

dx = 19
dy = 19

imgTypes = [".bmp"]

for root, dirs, files in os.walk("./test/Set14"):
    for afile in files:
        ffile = root + "/" + afile
        print('cutting', afile)

        if ffile[ffile.rindex("."):].lower() in imgTypes:
            im = Image.open(ffile)

            n = 1

            x = 0
            y = 0

            while x <= im.size[0] - dx:
                while y <= im.size[1] - dy:
                    name = "./test/14cut/" + afile.replace('.bmp', '_' + str(n) + '.bmp')
                    im2 = im.crop((y, x, y + dy, x + dx))
                    check = np.asarray(im2)
                    if not check.max() == 0:
                        im2.save(name)
                    y += dy
                    n += 1
                x += dx
                y = 0