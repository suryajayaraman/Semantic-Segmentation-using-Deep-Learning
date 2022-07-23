# basic imports
import os
import cv2
import numpy as np
from typing import Any, List
import matplotlib.pyplot as plt
from collections import OrderedDict

# DL library imports
import torch
from torch.utils.data import Dataset
from torchvision import transforms


trainTransform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
                ])


valTransform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
                ])


def getInverseTransform():
    inverse_transform = transforms.Compose([
            transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
        ])
    return inverse_transform



class cityScapeDataset(Dataset):
    def __init__(self, rootDir:str, folder:str, tf=None):
        """Dataset class for Cityscapes semantic segmentation data

        Args:
            rootDir (str): path to directory containing cityscapes image data
            folder (str) : 'train' or 'val' folder
        """        
        self.rootDir = rootDir
        self.folder = folder
        self.transform = tf

        # read rgb image list
        sourceImgFolder =  os.path.join(self.rootDir, 'leftImg8bit', self.folder)
        self.sourceImgFiles  = [os.path.join(sourceImgFolder, x) for x in sorted(os.listdir(sourceImgFolder))]

        # read label image list
        labelImgFolder =  os.path.join(self.rootDir, 'gtFine', self.folder)
        self.labelImgFiles  = [os.path.join(labelImgFolder, x) for x in sorted(os.listdir(labelImgFolder))]
    
    def __len__(self):
        return len(self.sourceImgFiles)
  
    def __getitem__(self, index):
        # read source image and convert to RGB, apply transform
        sourceImage = cv2.imread(f"{self.sourceImgFiles[index]}", -1)
        sourceImage = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            sourceImage = self.transform(sourceImage)

        # read label image and convert to torch tensor
        labelImage  = torch.from_numpy(cv2.imread(f"{self.labelImgFiles[index]}", -1)).long()
        return sourceImage, labelImage        




class cityScapeDataset_KD(Dataset):
    def __init__(self, rootDir:str, folder:str, tf=None):
        """Dataset class for Cityscapes semantic segmentation data

        Args:
            rootDir (str): path to directory containing cityscapes image data
            folder (str) : 'train' or 'val' folder
        """        
        self.rootDir = rootDir
        self.folder = folder
        self.transform = tf

        # reference folders for reading images
        sourceImgFolder =  os.path.join(self.rootDir, 'leftImg8bit', self.folder)
        labelImgFolder =  os.path.join(self.rootDir, 'gtFine', self.folder)
        teacherPredsFolder =  os.path.join(self.rootDir, 'teacherPreds', self.folder)

        # we're considering only files for which source image, label image 
        # and the teacher predictions are also available
        teacherPredsFiles = os.listdir(teacherPredsFolder)
        sourceImgFiles = os.listdir(sourceImgFolder)
        labelImgFiles = os.listdir(labelImgFolder)
        
        teacherPredsFilesBaseName = set([os.path.basename(x).split('.')[0] for x in teacherPredsFiles])
        sourceImgFilesBaseName = set([os.path.basename(x).split('.')[0] for x in sourceImgFiles])
        labelImgFilesBaseName = set([os.path.basename(x).split('.')[0] for x in labelImgFiles])

        commonFiles = sorted(list(teacherPredsFilesBaseName.intersection(sourceImgFilesBaseName).intersection(labelImgFilesBaseName)))
        print(len(commonFiles))
        
        
        # source image list
        self.sourceImgFiles  = [os.path.join(sourceImgFolder, f'{x}.png') for x in commonFiles]

        #  label image list
        self.labelImgFiles  = [os.path.join(labelImgFolder, f'{x}.png') for x in commonFiles]

        # teacher predictions file list
        self.teacherPredsFiles  = [os.path.join(teacherPredsFolder, f'{x}.pt') for x in commonFiles]
        
    def __len__(self):
        return len(self.sourceImgFiles)
  
    def __getitem__(self, index):
        # read source image and convert to RGB, apply transform
        sourceImage = cv2.imread(f"{self.sourceImgFiles[index]}", -1)
        sourceImage = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            sourceImage = self.transform(sourceImage)

        # read label image and convert to torch tensor
        labelImage  = torch.from_numpy(cv2.imread(f"{self.labelImgFiles[index]}", -1)).long()
        
        # read teacher predictions from file and return torch tensor
        teacherPreds = torch.load(self.teacherPredsFiles[index]).squeeze(0)
        return sourceImage, labelImage, teacherPreds

        



def displayDatasetSample(inputDataSet:Dataset, numSamples:int =2):
    """function displays sample image and label from input dataset
    Args:
        inputDataSet (Dataset): torch Dataset
        numSamples (KMeans): number of samples to display
    """    
    
    # predictions on random samples
    displaySamples = np.random.choice(len(inputDataSet), numSamples).tolist()
    fig, axes = plt.subplots(numSamples, 2, figsize=(2*4, numSamples * 3))
    inverse_transform = getInverseTransform()
 
    for i, sampleID in enumerate(displaySamples):
        sourceImage, labelImage = inputDataSet[sampleID]
        sourceImage = inverse_transform(sourceImage).permute(1, 2, 0).cpu().detach().numpy()
        labelImage = labelImage.cpu().detach().numpy()

        axes[i, 0].imshow(sourceImage)
        axes[i, 0].set_title("sourceImage")
        axes[i, 1].imshow(labelImage)
        axes[i, 1].set_title("labelImage")
    plt.show()


def visualizePredictions(model : torch.nn.Module, dataSet : Dataset,  
                         device :torch.device, numTestSamples : int ):
    """Function visualizes predictions of input model on samples from
    cityscapes dataset provided

    Args:
        model (torch.nn.Module): model whose output we're to visualize
        dataSet (Dataset): dataset to take samples from
        device (torch.device): compute device as in GPU, CPU etc
        numTestSamples (int): number of samples to plot
    """

    inverse_transform = getInverseTransform()
    model.to(device=device)

    # predictions on random samples
    testSamples = np.random.choice(len(dataSet), numTestSamples).tolist()
    _, axes = plt.subplots(numTestSamples, 3, figsize=(3*6, numTestSamples * 6))
    
    model.eval()
    for i, sampleID in enumerate(testSamples):
        inputImage, gt = dataSet[sampleID]
        inputImage = inputImage.to(device)

        y_pred = model(inputImage.unsqueeze(0))
        if(isinstance(y_pred, OrderedDict) == True):
            y_pred = y_pred['out']
        y_pred = torch.argmax(y_pred, dim=1).squeeze(0)

        landscape = inverse_transform(inputImage).permute(1, 2, 0).cpu().detach().numpy()
        # print(inputImage.shape, y_pred.shape, gt.shape, landscape.shape)
        label_class = gt.cpu().detach().numpy()
        label_class_predicted = y_pred.cpu().detach().numpy()
        axes[i, 0].imshow(landscape)
        axes[i, 0].set_title("Landscape")
        axes[i, 1].imshow(label_class)
        axes[i, 1].set_title("Label Class")
        axes[i, 2].imshow(label_class_predicted)
        axes[i, 2].set_title("Label Class - Predicted")
    plt.show()






class meanIoU:
    """
    Class to find the mean IoU using confusion matrix approach
        CFG (Any): object containing num_classes 
        device (torch.device): compute device
    """    
    def __init__(self, CFG, device):
        self.iouMetric = 0.0
        self.numClasses = CFG.NUM_CLASSES
        self.ignoreIndex = CFG.IGNORE_INDEX

        # placeholder for confusion matrix on entire dataset
        self.confusion_matrix = np.zeros((self.numClasses, self.numClasses))


    def update(self, y_preds: torch.Tensor, labels: torch.Tensor):
        """ Function finds the IoU for the input batch

        Args:
            y_preds (torch.Tensor): model predictions
            labels (torch.Tensor): groundtruth labels        
        Returns
        """
        predictedLabels = torch.argmax(y_preds, dim=1)
        batchConfusionMatrix = self._fast_hist(labels.numpy().flatten(), predictedLabels.numpy().flatten())
        # add batch metrics to overall metrics
        self.confusion_matrix += batchConfusionMatrix

    
    def _fast_hist(self, label_true, label_pred):
        """ function to calculate confusion matrix on single batch """
        mask = (label_true >= 0) & (label_true < self.numClasses)
        hist = np.bincount(
            self.numClasses * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.numClasses ** 2,
        ).reshape(self.numClasses, self.numClasses)
        return hist


    def compute(self):
        """ Returns overall accuracy, mean accuracy, mean IU, fwavacc """ 
        hist = self.confusion_matrix
        # acc = np.diag(hist).sum() / hist.sum()
        # acc_cls = np.diag(hist) / hist.sum(axis=1)
        # acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        # cls_iu = dict(zip(range(self.n_classes), iu))
        # return {"Overall Acc": acc, "Mean Acc": acc_cls, "Mean IoU": mean_iu, "Class IoU": cls_iu}
        return mean_iu

    def reset(self):
        self.iouMetric = 0.0
        self.confusion_matrix = np.zeros((self.numClasses, self.numClasses))
