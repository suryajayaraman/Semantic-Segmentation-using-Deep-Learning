# basic imports
import os
import cv2
import sys
import time
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import namedtuple


# DL library imports
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# import profiling package
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/pytorch-OpCounter-master')
from thop import profile
from thop import clever_format


NUM_CLASSES = 19
IGNORE_INDEX = 255


# when using torch datasets we defined earlier, the output image
# is normalized. So we're defining an inverse transformation to 
# transform to normal RGB format
inverse_transform = transforms.Compose([
        transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
    ])



# Based on https://github.com/mcordts/cityscapesScripts
CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                 'has_instances', 'ignore_in_eval', 'color'])
classes = [
    CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
]

train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color.append([0, 0, 0])
train_id_to_color = np.array(train_id_to_color)
id_to_train_id = np.array([c.train_id for c in classes])


def decodeTarget(target):
    target[target == 255] = 19
    return train_id_to_color[target]


def set_seed(seed : int):
    """Function to make results reproducible
    Args:
        seed (int): input seed
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True




def displayDatasetSample(inputDataSet, numSamples):    
    # select random samples from input dataset
    displaySamples = np.random.choice(len(inputDataSet), numSamples).tolist()

    # figure handle to plot
    fig, axes = plt.subplots(numSamples, 2, figsize=(3*4, numSamples * 3))
    if(numSamples == 1):
      axes = np.expand_dims(axes, axis=0)

    # iterate through samples 
    for i, sampleID in enumerate(displaySamples):
        sourceImage, labelImage = inputDataSet[sampleID]
        sourceImage = inverse_transform(sourceImage).permute(1, 2, 0).cpu().detach().numpy()
        labelImage = labelImage.cpu().detach().numpy()
        colorizedLabelImages = decodeTarget(labelImage).astype('uint8')
        axes[i, 0].imshow(sourceImage)
        axes[i, 0].set_title("sourceImage")
        axes[i, 1].imshow(colorizedLabelImages)
        axes[i, 1].set_title("labelImage")
    plt.show()




def visualizePredictions(model, dataSet, device, numTestSamples=2):
    """Function visualizes predictions of input model on samples from
    cityscapes dataset provided

    Args:
        model (torch.nn.Module): model whose output we're to visualize
        dataSet (Dataset): dataset to take samples from
        device (torch.device): compute device as in GPU, CPU etc
        numTestSamples (int): number of samples to plot
    """

    # ensure model is set to inference mode, moved to device
    model.eval();    
    model.to(device)

    # select random samples from input dataset
    testSamples = np.random.choice(len(dataSet), numTestSamples).tolist()
    _, axes = plt.subplots(numTestSamples, 3, figsize=(3*6, numTestSamples * 6))
    if(numTestSamples == 1):
        axes = np.expand_dims(axes, axis=0)

    # iterate through samples     
    for i, sampleID in enumerate(testSamples):
        inputImage, gt = dataSet[sampleID]
        inputImage = inputImage.to(device)

        # predict on input image
        y_pred = torch.argmax(model(inputImage.unsqueeze(0)), dim=1).squeeze(0)

        # plot RGB image
        rgb_image = inverse_transform(inputImage).permute(1, 2, 0).cpu().detach().numpy()
        axes[i, 0].imshow(rgb_image)
        axes[i, 0].set_title("RGB image")

        # Groundtruth label 
        gt_label = gt.cpu().detach().numpy()
        colorized_Gt_label = decodeTarget(gt_label).astype('uint8')
        axes[i, 1].imshow(colorized_Gt_label)
        axes[i, 1].set_title("GroundTruth Label")

        # Model predictions
        predicted_class = y_pred.cpu().detach().numpy()
        colorized_predictions = decodeTarget(predicted_class).astype('uint8')
        axes[i, 2].imshow(colorized_predictions)
        axes[i, 2].set_title("Model Predictions")
    plt.show()




def getModelPredInfo(model : nn.Module, input: torch.Tensor, device: torch.device):
    startTime = time.time()
    preds = model(input)
    endTime = time.time()
    
    # calculate model inference time in terms of FPS
    inferenceTime = endTime - startTime
    modelFPS = (1.0 / inferenceTime)
    return preds, modelFPS



def convertPredsToColorImg(preds : torch.Tensor, decodeFunction):
    predictedClasses_np = torch.argmax(preds, dim=1).cpu().detach().numpy()[0]
    colorizedPredictions = decodeTarget(predictedClasses_np).astype('uint8')
    return colorizedPredictions



def evaluteOnTestData(model : nn.Module, pretrainedModelPath:str, device : torch.device, 
                     dataloader_test : DataLoader, metricClass, metricName : str,
                     modelName:str, verbose : bool =False) ->float:
    """Evaluate the model on test set

    Args:
        model (nn.Module): input model
        pretrainedModelPath (str): path of weight file
        device (torch.device): compute device such as GPU or CPU
        dataloader_test (DataLoader): test dataset
        metricClass : function / class that calculates metric b/w predicted and ground truth  
        metricName (str) : name of metric
        modelName (str): name of the model
        verbose (bool, optional): flag to print results. Defaults to False.

    Returns:
        testSetMetric(float): metric on test data
    """
    testSetMetric = 0.0

    if verbose == True:
        print("------------------------")
        print(f"Test Data Results for {modelName} using {str(device)}")
        print("------------------------")
    
    modelLoadStatus = False
    if pretrainedModelPath is not None:
        if os.path.isfile(pretrainedModelPath) == True:
            model.load_state_dict(torch.load(pretrainedModelPath, map_location=device))
            modelLoadStatus = True
    # no need to load model
    else:
        modelLoadStatus = True

    if modelLoadStatus == True:
        lenTestLoader = len(dataloader_test)
        model.to(device)
        # set to inference mode
        model.eval()
        metricObject = metricClass(device=device)

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader_test, total=lenTestLoader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                y_preds = model(inputs)
            
                # update batch metric information            
                metricObject.update(y_preds.cpu().detach(), labels.cpu().detach())

        # compute metric of test set predictions
        testSetMetric = metricObject.compute()
        
        if verbose == True:
            print(f'{modelName} has {testSetMetric} {metricName} on testData')
    else:
        print(f'Model cannot load state_dict from {pretrainedModelPath}')
    return testSetMetric




class cityScapeDataset(Dataset):
    def __init__(self, rootDir, folder, tf=None):
        """Dataset class for Cityscapes semantic segmentation data

        Args:
            rootDir (str): path to directory containing cityscapes image data
            folder (str) : 'train' or 'val' folder
            tf (optional): transformation to apply. Defaults to None
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

        
class meanIoU:
    """
    Class to find the mean IoU using confusion matrix approach
        CFG (Any): object containing num_classes 
        device (torch.device): compute device
    """    
    def __init__(self, device):
        self.iouMetric = 0.0
        self.numClasses = NUM_CLASSES
        self.ignoreIndex = IGNORE_INDEX

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
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        return mean_iu