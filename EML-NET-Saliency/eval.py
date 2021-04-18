# This is the evaluation code to output prediction using our saliency model.
#
# Author: Sen Jia
# Date: 09 / Mar / 2020
#
import argparse
import os
import pathlib as pl

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
import numpy
import cv2
import skimage.io as sio

import resnet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('model_path', type=pl.Path,
                    help='the path of the pre-trained model')
parser.add_argument('img_path', type=pl.Path,
                    help='the folder of salicon data')

parser.add_argument('--gpu', default='0', type=str,
                    help='The index of the gpu you want to use')
parser.add_argument('--size', default=(480, 640), type=tuple,
                    help='resize the input image, (640,480) is from the training data, SALICON.')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
imgName = ''

def normalize(img):
    img -= img.min()
    img /= img.max()

def main():
    global args

    preprocess = transforms.Compose([
        transforms.Resize(args.size),
	transforms.ToTensor(),
    ])

    model = resnet.resnet50(args.model_path).cuda()
    model.eval()
    pil_img = Image.open(args.img_path).convert('RGB')
    openImg = cv2.cvtColor(numpy.array(pil_img),cv2.COLOR_RGB2BGR)
    #pil_img.show()
    #cv2.imshow('image',openImg)
    #cv2.waitKey(0)
    b2,g2,r2 = cv2.split(openImg)
    minValue = cv2.min(cv2.min(r2,g2),b2)
    #cv2.imshow('image',minValue)
    #cv2.waitKey(0)
    img = preprocess(pil_img)
    #have to be PIL images cause preprocess using resize that's only available for pil
    r2 = preprocess(Image.fromarray(r2))
    g2 = preprocess(Image.fromarray(g2))
    b2 = preprocess(Image.fromarray(b2))
    minValue = preprocess(Image.fromarray(minValue))
    newImg = torch.cat((img,r2,g2,b2,minValue),0)


    #processed = preprocess(pil_img).unsqueeze(0).cuda()
    processed = newImg.unsqueeze(0).cuda()


    with torch.no_grad():
        pred_batch = model(processed)

    for img, pred in zip(processed, pred_batch):
        fig, ax = plt.subplots(1, 2)

        pred = pred.squeeze()
        normalize(pred)
        pred = pred.detach().cpu()
        #pred_path = "D:\Code\cis4900\EML-NET-Saliency\smaps\\"+args.img_path.stem + "_smap.png"
        #idk if _smap is suppose to be on the end

        if(imgName[-5] == '.'):
            pred_path = "D:\Code\cis4900\EML-NET-Saliency\\mit1003smaps\\"+imgName[:-5]+".png"
        elif(imgName[-4] == '.'):
            pred_path = "D:\Code\cis4900\EML-NET-Saliency\\mit1003smaps\\"+imgName[:-4]+".png"


        sio.imsave(pred_path, pred)
        #img = img.permute(1,2,0).cpu()

        #ax[0].imshow(img)
        '''
        ax[0].imshow(img[:,:,0:3])
        ax[0].set_title("Input Image")
        ax[1].imshow(pred)
        ax[1].set_title("Prediction")
        plt.show()
        '''

if __name__ == '__main__':
    directory = "D:\Code\cis4900\EML-NET-Saliency\ALLSTIMULI\\"

    args.size = (768,1024)

    for filename in os.listdir(directory):
        imgName = filename
        args.img_path = directory+filename
        main()