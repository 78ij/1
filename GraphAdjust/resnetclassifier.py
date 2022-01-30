from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import copy
import matplotlib.pyplot as plt

def val_model(model, dataloaders,criterion):
# Each epoch has a training and validation phase
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    i = 0
    ret=[]
    for inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        
        #print(F.softmax(outputs))
       # print(preds)
        #print(labels)
        # statistics
        ret.append(F.softmax(outputs).flatten())
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloaders.dataset)
    epoch_acc = running_corrects.double() / len(dataloaders.dataset)
    return ret
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run



if __name__ == '__main__':
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)


    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "./rendered/perturbed2"

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet"

    # Number of classes in the dataset
    num_classes = 2

    # Batch size for training (change depending on how much memory you have)
    batch_size = 1

    # Number of epochs to train for
    num_epochs = 15

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)

    # Print the model we just instantiated
    print(model_ft)
    print("Initializing Datasets and Dataloaders...")
    data_transforms = {
        'perturbed2': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # Create training and validation datasets
    image_datasets = datasets.ImageFolder(data_dir, data_transforms['perturbed2'])
    # Create training and validation dataloaders
    dataloaders_dict = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=False, num_workers=0)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(20):
        print(image_datasets.imgs[i])
    # Send the model to GPU
    

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    model_ft = torch.load('classifier.pth')
    model_ft = model_ft.to(device)

    # Train and evaluate
    ret = val_model(model_ft, dataloaders_dict, criterion)
    plt.figure(1)
    plt.axis('off')
    for i in range(20):
        name = './rendered/perturbed2/good/00004f89-9aa5-43c2-ae3c-129586be8aaa_MasterBedroom-5863_' + str(i) + '.png'
        for x in range(20):
            #print(name)
            if(image_datasets.imgs[x][0] == name): 
                print(x)
                if i % 2 == 0: continue
                plt.subplot(2,5,int(i / 2 + 1))
                plt.title(str(ret[x][1].item()))
                img=plt.imread(name)
                plt.imshow(img)
                break
    plt.show()

