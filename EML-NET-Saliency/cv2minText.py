from PIL import Image
import cv2
import torch


def main():
    img = cv2.imread('salicon\\images\\COCO_train2014_000000581797.jpg')
    img2 = Image.open('salicon\\images\\COCO_train2014_000000581797.jpg').convert('RGB')
    #print(img)
    #img2.show()
    b,g,r = cv2.split(img)
    b = torch.tensor(b)
    print(b)
    print(b)
    #print(g)
    r2,g2,b2 = Image.Image.split(img2)
    print(r2)
    #print(b , b2)


if __name__ == '__main__':
    main()