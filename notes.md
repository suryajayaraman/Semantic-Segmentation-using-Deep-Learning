- Image segmentation -> process of classifying each pixel in an image belonging to a certain class.
- Semantic segmentation = classifying each pixel belonging to a particular label. It doesn't different across different instances of the same object. For example if there are 2 cats in an image, semantic segmentation gives same label to all the pixels of both cats
- Instance segmentation = Instance segmentation differs from semantic segmentation in the sense that it gives a unique label to every instance of a particular object in the image. As can be seen in the image above all 3 dogs are assigned different colours i.e different labels. With semantic segmentation all of them would have been assigned the same colour.


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








