# Image Segmentation References
 
 This is a collection of references to related blog posts, git repositories, research papers, libraries explaining many of concepts explained in the course

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
- [Xception Net blog post](https://towardsdatascience.com/review-xception-with-depthwise-separable-convolution-better-than-inception-v3-image-dc967dd42568)


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

- **deeplabv3 Pytorch implementations**
    - [deeplabv3+ pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)
    - [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)
    - [deeplabv3 impl](https://github.com/fregu856/deeplabv3)    



- **Segformer Pytorch implementations**
    - [Segformer arXiv paper](https://arxiv.org/abs/2105.15203)
    - [Official pytorch](https://github.com/NVlabs/SegFormer)
    - [Hugging face Segformer example](https://huggingface.co/docs/transformers/model_doc/segformer)
    - [Medium post 1](https://towardsdatascience.com/implementing-segformer-in-pytorch-8f4705e2ed0e)
    - [Segformer and HRNet pytorch implementation](https://github.com/camlaedtke/segmentation_pytorch)
    - [segformer-pytorch github](https://github.com/rulixiang/segformer-pytorch)
    - [Segformer with pretrained weights](https://github.com/anibali/segformer)
    - [Niels Rogge colab tutorial](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/SegFormer/Fine_tune_SegFormer_on_custom_dataset.ipynb#scrollTo=TMYYJ7_do08a)
    - [Panoptic Segformer](https://github.com/zhiqi-li/Panoptic-SegFormer)
    


### LayerNorm vs BatchNorm
- [Normalization technqiues blog post](https://medium.com/techspace-usict/normalization-techniques-in-deep-neural-networks-9121bf100d8)
- [Math behind LayerNorm](https://stats.stackexchange.com/questions/474440/why-do-transformers-use-layer-norm-instead-of-batch-norm)
- [Pytorch Layernorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
- [Understanding LayerNorm arXiv paper](https://proceedings.neurips.cc/paper/2019/file/2f4fe03d77724a7217006e5d16728874-Paper.pdf)
- [LayerNorm wandbi example](https://wandb.ai/wandb_fc/LayerNorm/reports/Layer-Normalization-in-Pytorch-With-Examples---VmlldzoxMjk5MTk1)


## Visualizing networks
- [DeepAI Quickie YT video](https://www.youtube.com/watch?v=I3VnY01JpSo)
- [DeepAI Quickie git repo](https://github.com/EscVM/EscVM_YT/blob/master/Notebooks/2%20-%20PT1.X%20DeepAI-Quickie/pt_1_vit_attention.ipynb)
- [Visual guide to Attention Math of Intelligence YT video](https://www.youtube.com/watch?v=mMa2PmYJlCo&t=58s)
- [vision transformer visualize attention map kaggle page](https://www.kaggle.com/code/piantic/vision-transformer-vit-visualize-attention-map/notebook)
- [visualize attention map git repo issue](https://github.com/tczhangzhi/VisionTransformer-Pytorch/issues/1)
- [Vision transformer colab noteobok](https://colab.research.google.com/github/hirotomusiker/schwert_colab_data_storage/blob/master/notebook/Vision_Transformer_Tutorial.ipynb)
- [exploring explainability of vision transformer](https://jacobgil.github.io/deeplearning/vision-transformer-explainability)
- [vit explain git repo](https://github.com/jacobgil/vit-explain)
- [Visualizing and understanding patch interactions in vision transformer](https://arxiv.org/abs/2203.05922)
- [Dino library](https://github.com/facebookresearch/dino)
- [lucid rains vit pytorch](https://github.com/lucidrains/vit-pytorch)
- [pytorch image models visualization self attention code](https://github.com/rwightman/pytorch-image-models/discussions/1232)
- [Transformer Interpretability Beyond Attention Visualization](https://arxiv.org/pdf/2012.09838v1.pdf)
- [Transformer explainability git repo](https://github.com/hila-chefer/Transformer-Explainability)


### Other Segmentation references
- [Real time Sem Seg paperswithcode](https://paperswithcode.com/task/real-time-semantic-segmentation)
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
- ## [UVA DLC Advanced DL](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html#The-Transformer-architecture)

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


 ## Git repo references
 - [Francesco Saverio Zuppichini](https://github.com/FrancescoSaverioZuppichini)
 - [Jacob Gildenblat](https://github.com/jacobgil)