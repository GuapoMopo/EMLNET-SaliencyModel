# This file contains the dataloader used to read imagres and annotations from the SALICON dataset.
#
# Author : Sen Jia 
#

#Questions
#To use torch.cat((), 1) with the 1, the rgb split images need to be converted to RGB, is that necessary? newImg = torch.cat((img,r,g,b),1)
#if i convert them to rgb and use 1 then i get RuntimeError: Given groups=1, weight of size [64, 6, 7, 7], expected input[8, 3, 1920, 640] to have 6 channels, but got 3 channels instead
#so use dim=0 for torch.cat

import random

import torch.utils.data as data
#from PIL import Image
import PIL.Image
from scipy import io
import torch
import cv2
import numpy

def make_trainset(root):
    img_root = root / "images" #my folders
    fix_root = root / "fixations"
    map_root = root / "maps"
    #img_root = root / "images/train" #doc doesn't include train folder
    #fix_root = root / "fixations/train"
    #map_root = root / "maps/train"


    files = [f.stem for f in img_root.glob("*.jpg")]
    print(files)
    images = []

    for f in files:
        print(f)
        img_path = (img_root / f).with_suffix(".jpg")
        fix_path = (fix_root / f).with_suffix(".mat")
        map_path = (map_root / f).with_suffix(".png")
        images.append([img_path, fix_path, map_path])

    return images

def pil_loader(path):
    #return Image.open(path).convert('RGB')
    return PIL.Image.open(path).convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def map_loader(path):
    return PIL.Image.open(path).convert('L')
    #return Image.open(path).convert('L')

def mat_loader(path, shape):
    mat = io.loadmat(path)["gaze"]
    fix = []
    for row in mat:
        data = row[0].tolist()[2]
        for p in data:
            if p[0]<shape[0] and p[1]<shape[1]: # remove noise at the boundary.
                fix.append(p.tolist())
    return fix

class ImageList(data.Dataset):
    def __init__(self, root, transform=None, train=False,
                 loader=default_loader, mat_loader=mat_loader, map_loader=map_loader):


        imgs = make_trainset(root)
        if not imgs:
            raise(RuntimeError("Found 0 images in folder: " + str(root) + "\n"))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.train = train
        self.loader = loader
        self.map_loader = map_loader
        self.mat_loader = mat_loader

    def __getitem__(self, index):

        img_path, fix_path, map_path = self.imgs[index]

        img = self.loader(img_path)
        w, h = img.size
        fixpts = self.mat_loader(fix_path, (w, h))
        smap = self.map_loader(map_path)

        fixmap = self.pts2pil(fixpts, img)

        if self.train:
            if random.random() > 0.5:
                #img = img.transpose(Image.FLIP_LEFT_RIGHT)
                #smap = smap.transpose(Image.FLIP_LEFT_RIGHT)
                #fixmap = fixmap.transpose(Image.FLIP_LEFT_RIGHT)
                img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                smap = smap.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                fixmap = fixmap.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            openImg = cv2.cvtColor(numpy.array(img),cv2.COLOR_RGB2BGR)
            b2,g2,r2 = cv2.split(openImg)
            minValue = cv2.min(cv2.min(r2,g2),b2)
            img = self.transform(img)
            r2 = self.transform(r2)
            g2 = self.transform(g2)
            b2 = self.transform(b2)
            minValue = self.transform(minValue)
            newImg = torch.cat((img,r2,g2,b2,minValue),0)
            smap = self.transform(smap)
            fixmap = self.transform(fixmap)
            '''
            img = self.transform(img)
            r = self.transform(r)
            g = self.transform(g)
            b = self.transform(b)
            minData = []
            minData.append([])
            for i in range(480): 
                minData[0].append([])
                for j in range(640):
                   minData[0][i].append(min(r[0][i][j],g[0][i][j],b[0][i][j]).item())
            minData = torch.tensor(minData)
            newImg = torch.cat((img,r,g,b, minData),0)
            smap = self.transform(smap)
            fixmap = self.transform(fixmap)
        return newImg, fixmap, smap
        
        
        if self.transform is not None:
            img = self.transform(img)
            smap = self.transform(smap)
            fixmap = self.transform(fixmap)'''
        return newImg, fixmap, smap
        
    def pts2pil(self, fixpts, img):
        #fixmap = Image.new("L", img.size)
        fixmap = PIL.Image.new("L", img.size)
        for p in fixpts:
            fixmap.putpixel((p[0], p[1]), 255)
        return fixmap

    def __len__(self):
        return len(self.imgs)

