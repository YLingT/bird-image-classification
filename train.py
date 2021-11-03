import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
import copy
import pickle
from efficientnet_pytorch import EfficientNet
import matplotlib.pylab as plt

# create dataset


class image_dataset(Dataset):
    def __init__(self, csvf, rootPath, transform):
        self.df = pd.read_csv(csvf, delimiter=' ',
                              header=None, names=['img', 'label-class'])
        self.df['label'] = self.df['label-class'].map(lambda x:
                                                      int(x.split('.')[0])-1)
        self.rootPath = rootPath
        self.xTrain = self.df['img']
        self.yTrain = self.df['label']
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.rootPath, self.xTrain[index]))
        img = img.convert('RGB')
        label = self.yTrain[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.xTrain.index)

    def __getlabels__(self):
        return self.yTrain

    def __classnum__(self):
        total_label = self.yTrain
        return len(np.unique(total_label))


train_transform = transforms.Compose([
    transforms.Resize((528, 528)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

train_set = image_dataset(csvf='./2021VRDL_HW1_datasets/training_labels.txt',
                          rootPath='./2021VRDL_HW1_datasets/training_images',
                          transform=train_transform)
train_loader = DataLoader(
    dataset=train_set, batch_size=4, shuffle=True, num_workers=16)

dataset_sizes = len(train_loader)
label = 200

print('Number of data')
print('========================================')
print('train size:: ', dataset_sizes, ' images')
print('class number:: ', label, ' species')
print('========================================')


def save_pickle(pkl_object, fname):
    pkl_file = open(fname, 'wb')
    pickle.dump(pkl_object, pkl_file)
    pkl_file.close()


def save_model_full(model, PATH):
    torch.save(model, PATH)


def train_model(model, criterion, optimizer, scheduler, device,
                train_dataloader, dataset_sizes,
                num_epochs=20, return_history=False,
                log_history=True, working_dir='output'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'epoch': [], 'train_loss': [], 'train_acc': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step(epoch)

        epoch_loss = running_loss/len(train_loader.dataset)
        epoch_acc = running_corrects.double()/len(train_loader.dataset)

        history['epoch'].append(epoch)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        print('lr: {}'.format(scheduler.optimizer.param_groups[0]['lr']))
        print('train Loss: {:.4f} train Acc: {:.4f}'.format(
              epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if log_history:
            save_pickle(history,
                        os.path.join(working_dir, 'model_history.pkl'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    print('Returning object of best model.')
    model.load_state_dict(best_model_wts)

    # save model
    save_model_full(model=model,
                    PATH=os.path.join(working_dir, 'EfficientNetb6_full.pth'))

    if return_history:
        return model, history
    else:
        return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device::', device)

model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=200)
model = model.to(device)

# set parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999),
                       weight_decay=0, eps=1e-08, amsgrad=False)
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                  factor=0.7, patience=3)

working_dir = r'./save_model'
if not os.path.exists(working_dir):
    os.makedirs(working_dir, exist_ok=True)
model_ft = train_model(model=model, criterion=criterion, optimizer=optimizer,
                       scheduler=exp_lr_scheduler, device=device,
                       train_dataloader=train_loader,
                       dataset_sizes=dataset_sizes,
                       num_epochs=2, working_dir=working_dir)


def unpickle(fname):
    file = open(fname, 'rb')
    obj = pickle.load(file)
    file.close()
    return obj

# Load model training history
model_history = './save_model/model_history.pkl'
history = unpickle(model_history)

plt.figure(figsize=(5, 5))
plt.subplot(1, 1, 1)
plt.plot(np.arange(0, np.max(history['epoch'])+1, 1),
         history['train_loss'], 'b-', label='Train')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss - Caltech Birds - EfficientNet')
plt.legend()

plt.show()
