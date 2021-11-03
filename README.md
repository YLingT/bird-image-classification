# bird-image-classification

This is the code for bird image classification on pytorch.

## Enviroment setting and dependencies 
Use pip install or conda install :
```
conda create --name test python=3.7.11
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install efficientnet_pytorch==0.6.3
pip install pandas==1.1.3
pip install matplotlib==3.4.3
```
And check the version :
```
#Name                        Version
python                       3.7.11
torch                        1.7.0
torchvision                  0.8.1
pandas                       1.1.3
pillow                       8.4.0
matplotlib                   3.4.3
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
criterion          cross entropy
optimizer          Adam
lr scheduler       ReduceLROnPlateau
```
Run code:
```
python train.py
```
Trained model will save in **save_model** directory.

### 2.  Testing and generate answer.txt file
```
python test.py
```
The answer.txt will save in **test_model** directory.

### 3.  Pre-trained models
Download trained model:  [link](https://drive.google.com/file/d/1pR8LjDpP9aigj5fw9rSioQ8-Pk3vu5D3/view?usp=sharing)  
Train Efficientnet-b6 model for 20 epoch, and the testing predict accuracy achieve 0.80547.    
 
Put the **EfficientNetb6_full.pth** in **save_model** directory and run test.py code.

<img height=300 src="https://github.com/YLingT/bird-image-classification/blob/main/training_accuracy.png"><img height=300 src="https://github.com/YLingT/bird-image-classification/blob/main/training_loss.png">
