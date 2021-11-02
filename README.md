# bird-image-classification

This is the code for bird image classification on pytorch.

## Enviroment setting and dependencies 
Use pip install or conda install :
* python==3.7
* torch==1.7.0+cu110
* torchvision==0.8.1+cu110
* panda==1.1.3
* pillow==8.4.0
* efficientnet-pytorch==0.6.3

## Dataset 
There are 200 species in provided dataset, 3000 images for training and 3033 images for testing.

## Code 
1. Data preparing
2. Training and validation
3. Testing and generate submission .txt file
4. Pre-trained models

## Result
Use the Efficientnet-b6 model and testing predict accuracy achieve 0.80547
