import os
import torch
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
from PIL import Image


def load(path):
    with Image.open(path) as img:
        x = ToTensor()(img)
    return x.unsqueeze(0)


imgs = []

path = './images'
        


for d in os.listdir(path):
    d = os.path.join(path, d)
    cs = os.listdir(d)
    cs.sort()

    path_gt = os.path.join(d, cs[0])
    a = [f for f in os.listdir(os.path.join(d, cs[0])) if 'target_origin'in f][0]
    path_gt = os.path.join(path_gt,a)

    for c in cs:
        c = os.path.join(d, c)
        f = [f for f in os.listdir(c) if 'n2' in f][0]
        target = os.path.join(c, f)
        imgs.append(load(target))

    imgs.append(load(path_gt))


x = torch.cat(imgs, dim=0)

grid =make_grid(x, 6)
img = ToPILImage()(grid)
#img.show()
img.save('./qualitative.png')
