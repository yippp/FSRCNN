class PairRandomCrop:
    image_crop_position = {}

    def __init__(self, size):
        import random
        import os
        import numbers

        from PIL import ImageOps

        self.os = os
        self.random = random
        self.ImageOps = ImageOps

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        pid = self.os.getpid()
        if pid in self.image_crop_position:
            x1, y1 = self.image_crop_position.pop(pid)
        else:
            x1 = self.random.randint(0, w - tw)
            y1 = self.random.randint(0, h - th)
            self.image_crop_position[pid] = (x1, y1)
        return img.crop((x1, y1, x1 + tw, y1 + th))
