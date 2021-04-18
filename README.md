# CIS4900
The EML-NET-Saliency code is completely Sen Jia's and the original can be found here https://github.com/SenJia/EML-NET-Saliency. I simply modified some sections to allow 7 input channels.

resnet.py
    Line 93: changed in_channels from 3 to 7
SaliconLoader.py
    Line 113-124: Added the code to increase data input from an image
eval.py
    - Overwrote the directory to pull images from as well as the size to resize them too
    - Added the same code from SaliconLoader to take more data from an image
    - Line 87: changed pred_path to my output path

