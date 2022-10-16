# thinkautonomous_imageSegmentation
Repo contains content created a for `IMAGE SEGMENTATION COURSE` offered at [thinkautonomous.ai](https://courses.thinkautonomous.ai/image-segmentation). This post is a gist of what the course teaches for anyone willing to learn about <u>Semantic Segmentation using Modern Deep Learning</u>

<span style="background-color:grey">Semantic Segmentation</span> <span style="background-color:grey">Pytorch</span> <span style="background-color:grey">Deeplabv3+</span>

[![Driveable Area Segmentation in Paris](images/Driveable_area_segmentation_Paris.gif)](https://www.youtube.com/watch?v=M6b9pjjvFw0 "Driveable Area Segmentation in streets of Paris")

## Agenda
- [Problem Statement](#problem-statement)
- [Deep Learning Project Components](#deep-learning-project-components)
- [Dataset]()
- [Loss function]()
- [Metric]()
- [Model]()
- [HyperParameters]()
- [Results]()
    - [Model wise comparison](#model-wise-comparison)
    - [Costa Rica Challenge](#costa-rica-challenge)

- [Inference Strategy](#inference-strategy)
- [Annexure]()
    - [References](references.md)
    - [Applications](#applications)
    - [Semantic vs Instance Segmentation](#semantic-vs-instance-segmentation)

## Problem Statement
- **Multi-class Segmentation Problem** - specifically to classify each pixel in an image to one of following 3 classes:
    - <span style="color:red">Direct / current lane (label 0)</span>
    - <span style="color:skyblue">Alternative lane (label 1)</span>     
    - <span style="color:black">Background (label 2)</span>

![problem statement](images/presentation/problem_statement.PNG)


## Deep Learning Project Components
- Following are the key areas of focus when trying to solve any Deep Learning Project PoC 

![Deep_learning_project_components](images/presentation/Deep_learning_project_components.png)


## Dataset
- We use the `Driveable Area` segment from [BDD100K dataset](https://www.bdd100k.com/) dataset for our project
- 3k labeled images split randomly into train, validation and test images(2.1k, 0.6k and 0.3k) 
- 

- It's a diverse dataset containing > 1000 hours of annotations for **Multi-Task Learning** tasks like Object Detection, Semantic, Instance Segmentation

## Results

### Model wise comparison


### Costa Rica Challenge
[![Driveable Area Segmentation in Costa Rica](images/Driveable_area_segmentation_Costa_Rica.gif)](https://www.youtube.com/watch?v=kpRdwTbuRxs "Driveable Area Segmentation in streets of Costa Rica")




## Inference Strategy

![Semantic Segmentation idea](images/presentation/semantic_segmentation_idea.png)
[Image source](https://www.researchgate.net/figure/Semantic-segmentation-of-a-scene-from-the-Cityscapes-dataset-by-Cordts-et-al-2016_fig24_316270100)



### Applications
- Semantic segmentation can be applied wherever Image and Image-like data is available, hence has numerous use-cases some of which are highlighted below

![Semantic Segmentation Applications](images/presentation/Semantic_segmentation_applications.png)


[Image source](https://keymakr.com/blog/semantic-segmentation-uses-and-applications/)


### Semantic vs Instance Segmentation
- Semantic Segmentation doesn't differentiate across different instances of the same object while Instance Segmentation does


![Semantic vs Instance Segmentation](images/presentation/instance_vs_semantic_segmentation.png)






[![Segformer MiT B3 Semantic Segmentation](images/Segformer_MiT_B3_Cityscapes_semantic_segmentation.gif)](https://youtu.be/NH4xPbxaXAY "Semantic Segmentation Cityscapes using Segformer MiT B3")


[![Segformer MiT B3 Attention Head Visualize](images/Segformer_MiT_B3_Cityscapes_Attention_Head_visualize.gif)](https://www.youtube.com/watch?v=BG8MoGAYMkA "Segformer-MiT-B3 Attention heads visualization on Cityscapes dataset")