# thinkautonomous_imageSegmentation
Repo contains content created a for `IMAGE SEGMENTATION COURSE` offered at [thinkautonomous.ai](https://courses.thinkautonomous.ai/image-segmentation). This post is a gist of what the course teaches for anyone willing to learn about <u>Semantic Segmentation using Modern Deep Learning</u>

## Agenda
- [Problem Statement](#problem-statement)
- [Deep Learning Project Components](#deep-learning-project-components)


- [Inference Strategy](#inference-strategy)
- [Applications](#applications)
- [Semantic vs Instance Segmentation](#semantic-vs-instance-segmentation)

## Problem Statement
- **Multi-class Segmentation Problem** - specifically to classify each pixel in an image to one of following 3 classes:
    - Direct / current lane (label = 0)
    - Alternative lane (label = 1)
    - Background (label = 2)

![problem statement](images/presentation/problem_statement.PNG)


## Deep Learning Project Components
- Following are the key areas of focus when trying to solve any Deep Learning Project PoC 

![Deep_learning_project_components](images/presentation/Deep_learning_project_components.png)


## Inference Strategy

![Semantic Segmentation idea](images/presentation/semantic_segmentation_idea.png)
[Image source](https://www.researchgate.net/figure/Semantic-segmentation-of-a-scene-from-the-Cityscapes-dataset-by-Cordts-et-al-2016_fig24_316270100)



## Applications
- Semantic segmentation can be applied wherever Image and Image-like data is available, hence has numerous use-cases some of which are highlighted below

![Semantic Segmentation Applications](images/presentation/Semantic_segmentation_applications.png)


[Image source](https://keymakr.com/blog/semantic-segmentation-uses-and-applications/)


## Semantic vs Instance Segmentation
- Semantic Segmentation doesn't differentiate across different instances of the same object while Instance Segmentation does


![Semantic vs Instance Segmentation](images/presentation/instance_vs_semantic_segmentation.png)




[![Driveable Area Segmentation in Paris](images/Driveable_area_segmentation_Paris.gif)](https://www.youtube.com/watch?v=M6b9pjjvFw0 "Driveable Area Segmentation in streets of Paris")


[![Driveable Area Segmentation in Costa Rica](images/Driveable_area_segmentation_Costa_Rica.gif)](https://www.youtube.com/watch?v=kpRdwTbuRxs "Driveable Area Segmentation in streets of Costa Rica")


[![Segformer MiT B3 Semantic Segmentation](images/Segformer_MiT_B3_Cityscapes_semantic_segmentation.gif)](https://youtu.be/NH4xPbxaXAY "Semantic Segmentation Cityscapes using Segformer MiT B3")


[![Segformer MiT B3 Attention Head Visualize](images/Segformer_MiT_B3_Cityscapes_Attention_Head_visualize.gif)](https://www.youtube.com/watch?v=BG8MoGAYMkA "Segformer-MiT-B3 Attention heads visualization on Cityscapes dataset")