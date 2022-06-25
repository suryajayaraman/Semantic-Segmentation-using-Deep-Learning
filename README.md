# thinkautonomous_imageSegmentation
Repo containing content for thinkautonomous image segmentation course 

- Image segmentation -> process of classifying each pixel in an image belonging to a certain class.
- Semantic segmentation = classifying each pixel belonging to a particular label. It doesn't different across different instances of the same object. For example if there are 2 cats in an image, semantic segmentation gives same label to all the pixels of both cats
- Instance segmentation = Instance segmentation differs from semantic segmentation in the sense that it gives a unique label to every instance of a particular object in the image. As can be seen in the image above all 3 dogs are assigned different colours i.e different labels. With semantic segmentation all of them would have been assigned the same colour.



## Types of convolutions
1. [Medium post on different types of convolutions in DL](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)
2. [Separable convolutions intro](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)
3. [Arithmetic on convolutions arXiv paper](https://arxiv.org/pdf/1603.07285.pdf)
4. [NPTEL lecture on Atrous convolutions and Transposed convolutions](https://www.youtube.com/watch?v=gmr18xg4wTg)
5. [Pytorch Upsample docs](https://pytorch.org/docs/stable/generated/torch.nn.Upsample)
6. [torch nn functional interpolate](https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)
7. [Deconvolution and checkerboard artifacts blog post](https://distill.pub/2016/deconv-checkerboard/)
8. [How to use upsampling in pytorch gitpost](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-upsample-for-upsampling-with-pytorch.md)
9. [Upsampling SO post](https://stackoverflow.com/questions/64284755/what-is-the-upsampling-method-called-area-used-for)
10. [Functional.interpolate vs nn.Upsample Pytorch forum](https://discuss.pytorch.org/t/which-function-is-better-for-upsampling-upsampling-or-interpolate/21811/7)



### Research papers
- [Image Segmentation Using Deep Learning: A Survey](https://arxiv.org/abs/2001.05566)
- [Comparison of Semantic Segmentation for Remote sensing](https://arxiv.org/abs/1905.10231)
- [Multi scale contex using Dilated convolutions](https://arxiv.org/abs/1511.07122)
- [PAN paper](https://arxiv.org/pdf/1906.04378.pdf)


### Blog posts
- [Overall intro to semantic segmentation](https://nanonets.com/blog/semantic-image-segmentation-2020/)
- [neptune ai blog post](https://neptune.ai/blog/image-segmentation)
- [Stanford Lecture video](https://www.youtube.com/watch?v=nDPWywWRIRo)
- [Stanford lecture notes](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf)
- [topbots semseg blog post](https://www.topbots.com/semantic-segmentation-guide/)
- [Tips for Segmentation and object detection](https://neptune.ai/blog/data-exploration-for-image-segmentation-and-object-detection)
- [Kaggle tips and tricks](https://neptune.ai/blog/image-segmentation-tips-and-tricks-from-kaggle-competitions)
- [jeremy jordan blog post](https://www.jeremyjordan.me/semantic-segmentation/)
- [CVPR 2022 papers summary](https://github.com/Jeremy26/CVPR-2022-Papers-EN/blob/main/README.md#2)
- [PAN blog post](https://medium.com/mlearning-ai/review-pan-pyramid-attention-network-for-semantic-segmentation-semantic-segmentation-8d94101ba24a)


### Code implementations
- ***[pytorch segmentation library](https://github.com/yassouali/pytorch-segmentation)***
- ***[segmentation models pytorch](https://github.com/qubvel/segmentation_models.pytorch)***
- ***[pytorch-semseg library](https://github.com/meetps/pytorch-semseg)***
- ***[sssegmentation pytorch library](https://github.com/SegmentationBLWX/sssegmentation)***
- ***[libtorch segmentation C++ library](https://github.com/AllentDan/LibtorchSegmentation)***



- **UNet Pytorch implementations**
    - [link 1](https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py)
    - [link 2](https://amaarora.github.io/2020/09/13/unet.html)
    - [link 3](https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py)
    - [link 4](https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-model-building-6ab09d6a0862)

- **PSPNET Pytorch implementations**
    - [Author Official pytorch implementation](https://github.com/hszhao/semseg/blob/master/model/pspnet.py)
    - [Lextal implementation with pretrained backbones](https://github.com/Lextal/pspnet-pytorch)
    - [IanTaehoon implementation](https://github.com/IanTaehoonYoo/semantic-segmentation-pytorch/blob/master/segmentation/models/pspnet.py)


- [Real time Sem Seg paperswithcode](https://paperswithcode.com/task/real-time-semantic-segmentation)
- [deeplabv3 impl](https://github.com/fregu856/deeplabv3)
- [deeplabv3+ pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)
- [deeplabv3 r101](https://www.kaggle.com/mobassir/deeplabv3-resnet101-for-severstal-sdd)
- [HRNet official implementation](https://github.com/HRNet/HRNet-Semantic-Segmentation)
- [HRNet Kaggle SemSeg post](https://www.kaggle.com/bibek777/try-hrnet-semantic-segmentation)
- [Lyft perception challenge 4th place soln](https://github.com/NikolasEnt/Lyft-Perception-Challenge)
- [Github post Awesome semantic segmentation](https://github.com/mrgloom/awesome-semantic-segmentation)
- [Mask RCNN implementation for BDD100k](https://github.com/TilakD/Object-detection-and-segmentation-for-self-driving-cars)


### Transformer architectures for segmentation
- [segmenter github](https://github.com/rstrudel/segmenter)
- [UTNet](https://github.com/yhygao/UTNet)
- [transUTNet](https://github.com/mkara44/transunet_pytorch)
- [Hugging face transformer Segformer for Semantic segmentation](https://huggingface.co/blog/fine-tune-segformer)


### Semantic segmentation metrics
- [pytorch official impl](https://github.com/pytorch/vision/blob/main/references/segmentation/utils.py)
- [kaggle post on semSeg metrics](https://www.kaggle.com/yassinealouini/all-the-segmentation-metrics#Before-you-go)
- [metrics blog post](https://www.jeremyjordan.me/evaluating-image-segmentation-models/)
- [metrics blog post 2](https://ilmonteux.github.io/2019/05/10/segmentation-metrics.html)
- [torch ignite implementation](https://github.com/pytorch/ignite/blob/master/ignite/metrics/metric.py)
- [fastai forum post](https://forums.fast.ai/t/multi-class-semantic-segmentation-metrics-and-accuracy/74665/3)
- [scikit learn jaccard score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html)
- [torchmetrics library](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#iou)


## Loss functions
- [SMP implementations](https://smp.readthedocs.io/en/latest/losses.html)
- [medium post](https://medium.com/@junma11/loss-functions-for-medical-image-segmentation-a-taxonomy-cefa5292eec0#:~:text=Generalized%20Dice%20loss%20is%20the,hard%20cases%20with%20low%20probabilities.)
- [IoU vs Dice](https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou/276144#276144)
- [survey of loss functions for semantic segmentation](https://arxiv.org/pdf/2006.14822.pdf)

- [Focal loss medium post](https://medium.com/swlh/understanding-focal-loss-for-pixel-level-classification-in-convolutional-neural-networks-720f19f431b1)

- [focal loss for segmentation blog post with code](https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c)

 - [Loss functions with pytorch implentations](https://github.com/JunMa11/SegLoss)

 - [Focal and Lovasz loss pytorch](https://github.com/Hsuxu/Loss_ToolBox-PyTorch)




 
 

## Use-cases of image segmentation
- Handwritten recognition
- Google portrait mode
- YouTube stories
- Virtual make-up
- Virtual try-on 
- Visual Image Search
- Self-driving cars

## Methods and Techniques

### 1. Fully Convolutional Network
    - FC layers replaced by conv layers
    - input image can be variable size
    - feature map is downsampled embedding of input image -> need to up-sample it using an interpolation technique. 
    - Bilinear up sampling works but paper proposes using learned up sampling with deconvolution which can even learn a non-linear up sampling.
    - Output = rough due to 32x reduction due to downsampling. Very difficult for the network to do 32x upsampling by using this little information. This architecture is called FCN-32
    - Soln : In FCN-16 information from the previous pooling layer is used along with the final feature map -> network has to learn 16x up sampling, better than FCN-32. FCN-8 tries to make it even better by including information from one more previous pooling layer.

### 2. Unet
    - Built on top of FCN for medical purposes to find tumours in lungs or the brain. 
    - Encoder decoder architecutre with feature map upsampling using learned deconvolution layers
    - Different from FCN that it adds shortcut connections b/w relevant encoder and decoder layers


### 3. DeepLab
    - Atrous convolutions
        - 32x downsampling cause huge information loss and 32x upsampling is computationally, memory expensive
        - Atrous convolution / hole convolution / Dilated convolution to solve problem
        - Dilated convolution increases filter size by appending zeros(called holes) to fill the gap between parameters. The number of holes/zeroes filled in between the filter parameters is called by a term dilation rate. When the rate = 1, normal convolution. When rate is equal to 2 one zero is inserted between every other parameter making the filter look like a 5x5 convolution. Now it has the capacity to get the context of 5x5 convolution while having 3x3 convolution parameters. Similarly for rate 3 the receptive field goes to 7x7.
        - Last pooling layers have stride = 1 instead of 2 -> down sampling rate = 8x. Series of atrous convolutions to capture the larger context. For training the output labelled mask is down sampled by 8x to compare each pixel. For inference, bilinear up sampling is used to produce output of the same size -> decent enough results at lower computational/memory costs since bilinear up sampling doesn't need any parameters as opposed to deconvolution for up sampling.

    - Atrous Spatial Pyramidal Pooling
        - Spatial Pyramidal Pooling introduced in SPPNet to capture multi-scale information from a feature map. Before, input images at different resolutions are supplied and the computed feature maps are used together to get the multi-scale information but this takes more computation and time. With Spatial Pyramidal Pooling multi-scale information can be captured with a single input image.
        - SPP module the network produces 3 outputs of dimensions 1x1(i.e GAP), 2x2 and 4x4. These values are concatenated by converting to a 1d vector thus capturing information at multiple scales. Another advantage of using SPP is input images of any size can be provided.
        - ASPP takes the concept of fusing information from different scales and applies it to Atrous convolutions. The input is convolved with different dilation rates and the outputs of these are fused together.
        - Input is convolved with 3x3 filters of dilation rates 6, 12, 18 and 24 and the outputs are concatenated together since they are of same size. 
        - A 1x1 convolution output is also added to the fused output. To also provide the global information, the GAP output is also added to above after up sampling. The fused output of 3x3 varied dilated outputs, 1x1 and GAP output is passed through 1x1 convolution to get to the required number of channels.

    - Conditional Random Fields usage for improving final output
        - Pooling reduces the number of parameters and brings invariance property (quality of a neural network being unaffected by slight translations in input). But invariance, segmentation output is coarse and the boundaries are not concretely defined.
        - Soln by using graphical model CRF, A post-processing step to define shaper boundaries. It works by classifying a pixel based not only on it's label but also based on other pixel labels.
    
    - `Deeplab-v3` introduced batch normalization and suggested dilation rate multiplied by (1,2,4) inside each layer in a Resnet block. Adding image level features to ASPP module was proposed as part of this paper


    - `Deeplab-v3+` suggested to have a decoder instead of plain bilinear up sampling 16x. Inspired by U-Net to take information from encoder layers to improve the results. The encoder output is up sampled 4x using bilinear up sampling and concatenated with the features from encoder which is again up sampled 4x after performing a 3x3 convolution. This approach yields better results than a direct 16x up sampling. Also modified Xception architecture is proposed to be used instead of Resnet as part of encoder and depthwise separable convolutions are now used on top of Atrous convolutions to reduce the number of computations.


## Metrics

- ### Pixel Accuracy
        - Accuracy = ratio of correctly classified pixels w.r.t total pixels
        - Accuracy = (TP+TN)/(TP+TN+FP+FN)
        - Works poorly in unbalanced datasets

- ### Intersection Over Union
        - IOU = Ratio of intersection of ground truth and predicted segmentation outputs over their union. 
        - For multiple classes, IOU of each class is calculated and their mean is taken.

- ### Frequency weighted IOU
        - Extension of IOU with weight for each class score

- ### F1 Score
        - 2PR / (P+R)

## Loss functions
- ### Cross Entropy Loss
    - Simple avg CE classification loss for every pixel, but it suffers due to class imbalance
    - FCN uses class weights
    - UNet gives more weightage to border pixels

- ### Focal Loss
    - 

## Upsampling techniques
- Nearest neighbour upsampling
- bilinear interpolation
- bed of nails approach
- max unpooling
- transposed convolutions
- deconvolutions