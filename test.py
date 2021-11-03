import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pickle
import zipfile


# build test dataset
class image_dataset(Dataset):
    def __init__(self, csf, rootPath, transform):
        header_list = ['img']
        self.df = pd.read_csv(csf, names=header_list)
        self.rootPath = rootPath
        self.xTrain = self.df['img']
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.rootPath, self.xTrain[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.xTrain.index)

test_transform = transforms.Compose([
    transforms.Resize((528, 528)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

test_set = image_dataset(csf='./2021VRDL_HW1_datasets/testing_img_order.txt',
                         rootPath='./2021VRDL_HW1_datasets/testing_images',
                         transform=test_transform)

test_loader = DataLoader(
    dataset=test_set, batch_size=1, shuffle=False, num_workers=4)

dataset_sizes = len(test_loader)
print('Number of data')
print('========================================')
print('test size:: ', dataset_sizes, ' images')
print('========================================')


def save_pickle(pkl_object, fname):
    pkl_file = open(fname, 'wb')
    pickle.dump(pkl_object, pkl_file)
    pkl_file.close()


def test_model(model, device, dataloaders,
               dataset_sizes, class_name, working_dir='output'):
    with open('./2021VRDL_HW1_datasets/testing_img_order.txt') as f:
        test_name = [x.strip() for x in f.readlines()]
    submission = []
    with torch.no_grad():
        model.eval()
        for i, inputs in enumerate(dataloaders):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            submission.append([test_name[i], class_name[preds.item()]])
            print("test_name=", test_name[i], ",",
                  " pred= ", class_name[preds.item()])

    np.savetxt(os.path.join(working_dir, 'answer.txt'), submission, fmt='%s')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device::', device)

# load model
output_dir = r'./save_model'
model_file = os.path.join(output_dir, 'EfficientNetb6_full.pth')
model = torch.load(model_file)

with open('./2021VRDL_HW1_datasets/classes.txt') as f:
    class_name = [x.strip() for x in f.readlines()]
working_dir = r'./test_model'
if not os.path.exists(working_dir):
    os.makedirs(working_dir, exist_ok=True)
model_ft = test_model(model=model, device=device,
                      dataloaders=test_loader, dataset_sizes=dataset_sizes,
                      class_name=class_name, working_dir=working_dir)
