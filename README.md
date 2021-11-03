# bird-image-classification

This is the code for bird image classification on pytorch.

## Enviroment setting and dependencies 
Use pip install or conda install :
```
#Name                        Version
python                       3.7.11
torch                        1.7.0+cu110
torchvision                  0.8.1+cu110
panda                        1.1.3
pillow                       8.4.0
efficientnet-pytorch         0.6.3
```
## Dataset 
There are 200 species in provided dataset, 3000 images for training and 3033 images for testing.

## Code 
### 0. Download Project
```
git clone https://github.com/YLingT/bird-image-classification
cd bird-image-classification
```
Extract the 2021VRDL_HW1_datasets.zip file, training_images.zip and the testing_images.zip in it.  
The project structure are as follows:
```
bird-image-classification
  |── 2021VRDL_HW1_datasets
      |── training_images
      |── testing_images
      |── training_labels.txt
      |── testing_img_order.txt
      |── classes.txt
  |── train.py
  |── test.py
```
### 1.  Training
Parameter setting:
```
epoch              20
batch size         4
learning rate      0.0001
optimizer          Adam
lr scheduler       ReduceLROnPlateau
```
Run code:
```
python train.py
```
Trained model will save in save_model file.
### 2.  Testing and generate submission .txt file
```
python test.py
```
### 3.  Pre-trained models
Use the Efficientnet-b6 model training for 20 epoch, and the testing predict accuracy achieve 0.80547.  
Training accuracy finally achieve 0.9963, and the final training loss is 0.0357.


