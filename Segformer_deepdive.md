# SEGFORMERS - Semantic Segmentation using Transformer Networks
Repo contains material from `SEGFORMERS COURSE` offered at [thinkautonomous.ai](https://courses.thinkautonomous.ai/image-segmentation). This post is a gist of what the course teaches for anyone willing to learn about <u>Attention mechanism, Transformers, Semantic Segmentation using Segformers</u>

[![Segformer MiT B3 Semantic Segmentation](images/Segformer_MiT_B3_Cityscapes_semantic_segmentation.gif)](https://youtu.be/NH4xPbxaXAY "Semantic Segmentation Cityscapes using Segformer MiT B3")


- This post is 2nd in series of Semantic Segmentation using Modern Deep Learning. [*Check out the 1st post to start with fundamental concepts of Semantic segmentation*](README.md)
- [*For people interested in Deployment strategies - here's a deepdive into Neural  Optimization techniques*](https://github.com/suryajayaraman/thinkAutonomous_modelOptimization/blob/main/README.md)
- [***CLICK HERE TO CHECK OUT ALL OF MY PROJECTS***](https://suryajayaraman.github.io/)



## Agenda
- [Problem Statement](#problem-statement)
- [Deep Learning Project Components](#deep-learning-project-components)
    - [Dataset](#dataset)
    - [Loss function](#loss-function)
    - [Metric](#metric)
- [Attention Mechanism](#attention-mechanism)
- [Transformers](#transformers)
    - [Self-Attention](#self-attention)
    - [Query, Key and Value](#query-key-and-value)
    - [Multi-Head Self-Attention](#multi-head-self-attention)
    - [Pros and Cons of Transformer](#pros-and-cons-of-transformer)
- [Vision Transformers](#vision-transformers)
    - [Intuition of ViT](#intuition-of-vit)
- [Segformers](#segformers)
    - [Overlap Patch Embedding](#overlap-patch-embedding)
    - [Efficient self-attention](#efficient-self-attention)
    - [Attention Intuition in images](#attention-intuition-in-images)
    - [Mix FFN](#mix-ffn)
    - [Encoder](#encoder)
    - [All MLP Decoder](#all-mlp-decoder)
- [HyperParameters](#hyperparameters)
- [Results](#results)
- [Visualizing Attention Mask](#visualizing-attention-mask)


## Problem Statement
- **Multi-class Segmentation Problem** - specifically to classify each pixel in an image to one of 19 classes:

![Sample picture statement](images/presentation/semantic_segmentation_idea.png)


## Deep Learning Project Components

## Dataset
We use the `Semantic Segmentation` segment from [Cityscapes dataset](https://www.cityscapes-dataset.com/) for our project
- 3.5k labeled images split randomly into train, validation and test images (2.38k, 0.595k and 0.5k). Input Image -> (512, 1024, 3) RGB image and labels -> (512, 1024) **uint8 datatype**
- As with most semantic segmentation datasets, there is data imbalance:

![cityscapes_class_distribution](images/presentation/cityscapes_class_distribution.PNG)

[Image reference](https://www.cityscapes-dataset.com/wordpress/wp-content/papercite-data/pdf/cordts2016cityscapes.pdf)

- **We'll need to account for class imbalance when selecting the loss function and metric**. As for pre-processing, No Data Augumentation was applied other than *Normalization using Imagenet mean, standard deviation*
- [Cityscapes scripts git repo](https://github.com/mcordts/cityscapesScripts) is the official place containing useful scripts to help you get started on the dataset.


## Loss function
As mentioned earlier, Loss function must be able to handle class imbalance. Personally, I expected Dice loss to perform better than normal Cross Entropy, but <u>CE loss was >> than Dice loss </u>
- [This post](https://stats.stackexchange.com/questions/321460/dice-coefficient-loss-function-vs-cross-entropy) proposes that 
Gradients are nicer in CE compared to Dice loss
- In hindsight, one can argue that with increased number of classes, Dice loss might result in *difficult to flow gradients* Hence, ***Cross Entropy loss is chosen as loss function***

## Metric
**mean IoU** is chosen as Evaluation Metric for 2 reasons:
- Imbalanced nature of dataset
- It's the <u>industry standanrd for most segmentation tasks</u>


## Attention Mechanism
- Attention mechanism originated in NLP domain. Traditionally, SOTA models at that time had Encoder-Decoder architecture which would contain sequence of LSTM / RNN modules
- For case of Machine Translation, Each word in input sentence would be parsed 1 by 1 and finally, <u> one single context vector is given to decoder to get output. But for very long sentences, this single context vector was not sufficient to remember relevant context for each word</u>


![Encoder_decoder_with_without_attn](images/presentation/Encoder_decoder_with_without_attn.png)

- This is where Attention mechanism was introduced. Instead of just giving a single context vector, for each output in decoder, the network was also given all the context from encoder and allowed to learn weights for each of them.
- **It allowed the network to learn which parts of input sentence to focus on when deciphering each word in decoder**

- So, Attention can be summarised as **giving different levels of importance (different weightage) to different parts of input**

![attention_concept](images/presentation/attention_concept.png)

- Assuming above image is input to object classification, to classify it as human, we may need to focus more on human attributes like the face, hands and less on the background
- This weighting mechanism is <u>learned during training</u> and enables to <u>understand context at each stage of model</u>
- It also helps in increasing <u>model interpretability</u>


## Transformers
- [Transformers](https://arxiv.org/abs/1706.03762) is a revolutionary architecture **solely based on Attention mechanism** introduced originally for NLP domain for tasks like Machine Translation, Language to speech etc

![transformer_explanation](images/presentation/original_transformer_arch.png)

- Apart from Attention modules, there 3 components - Embedding layers (Input and Output embeddings), Normalised Residual connection (Add & norm) and Fully Connected layer (Feed Forward). These components together with Attention module form the core of any Transformer block

- Core module of transformer is **Multi-Head Self Attention**. We'll break down each of component step by step

### Self-Attention
- Self-Attention is Attention applied on itself

![self_attention](images/presentation/self_attention.png)

- In 1st example shown above, one can see that **when parsing the word `it`, more weightage is given to `cat` indicating the context model has learnt**
- In the 2nd example, the model must learn that **1st bank refers to the financial institution and 2nd to river bank, hence context for both words are different**


### Query, Key and Value
- Attention is achieved through Query-Key-Value mechanism
- Consider example of searching for some content in YouTube. We would enter title of our choice and YouTube would return most relevant results in order of how close they are to the words we entered. Here,
    - Query is the words we entered
    - Key is the title associated with each video
    - Value is the video itself

![Query_Key_value](images/presentation/Query_Key_value.png)

- We're comparing Query and Key and ranking results based on how close / relevant they're to each other
- In Linear Algebra, **one way to check relevance of one vector to another is Dot Product / cosine similarity**
- On a crude level, Dot product b/w Query (the words we entered) and Key (title of each video) is calculated and output is normalized to probability and videos with highest weights are shown as top results
- Actual steps : We compute Dot product b/w Query and Key which is softmax to give attention weights. Output is then calculated using weighted sum of values
- **Note that Query, Key and Value which enable Attention are learned when training the model**

### Multi-Head Self-Attention
- Multi-head self attention is simple extension of Self Attention
- We create multiple heads (branches) from same input, with each having its own Query, Key, Value and finally, we combine results from all heads which is passed on to next stage
![Multi_head_self_attention](images/presentation/Multi_head_self_attention.png)
- Continuing our object classification example, one head might learn to segment human while another identifies greenery in background and another head tries to learn about sky etc
- This way, **different heads can focus on different perspectives to arrive at the result, providing more robust decision output**


### Pros and Cons of Transformer

<u>Pros</u>
- Transformers achieved better than SOTA metric compared to LSTM-variants while being ligter
- Its highly parallelizable architecture (>> better than LSTM and > than Convolution based models), enabling faster training
- Enabled Transfer Learning in NLP domain
- Much better Model Interpretability
- MLP layers offered better scope for Quantization and other optimization techniques compared to Convolution / LSTM modules

<u>Cons</u>
- Require huge amounts of data to realize the full potential of Transformer models
- They have larger memory footprint and slower in terms of absolute numbers


## Vision Transformers
- [Vision Transformers](https://arxiv.org/abs/2010.11929) is simply Transformers applied to image Input. 

![vision_transformer_arch](images/presentation/vision_transformer_arch.png)

- Images separated into patches and each patch ~= word in sentence
- Position encoding to represent order of sequence
- After Encoder, any module can be used as decoder according to  target Task.
- As with any Transformer model, we can visualize the attention masks to see what the model is focussing on 

[![Vision_transformer_Attention_masks](images/Vision_transformer_Attention_masks.gif)](https://www.youtube.com/watch?v=n_zB-nEa3AI "ViT Attention heads visualization on Cityscapes dataset")

- In the above vide, we can see each of the Attention head is focussing on different aspects. eg : Attention heads 4 & 5 are focussing on traffic lights and car headlights, Attention head 6 is focussing mostly on far away objects etc

- <u>Note </u> : Vision Transformer doesn't use multi-scale features. Hence Encoder output is mostly an downsampled version of the input image by some factor


## Segformers
- Segformer is an Encoder-decoder version of Vision Transformer, for segmentation tasks
- Multi-scale problem tackled by feeding encoder outputs at different stages to decoder
- Decoder is simple MLP which combines multi-scale encoder outputs

![segformer_corrected_architecture](images/presentation/segformer_corrected.png)

- As seen above, there are 3 main parts in Transformer block of Encoder - Overlap Patch Embedding, Efficient self-attention and Mix-FFN. We'll cover these modules one by one.

### Overlap Patch Embedding
- Entire image is split into patches which are passed to consequent Transformer blocks
- This layer acts like Embedding layer and uses Convolution layers
- Intuitively, the operations behind Overlap patch embedding is as follows :

![overlap_patch_embeddings](images/presentation/overlap_patch_embeddings.png)

- **The authors note that smaller patch sizes are preferrable to segmentation tasks (7 for 1st stage and 3 for later stages) and need for overlapping patches to inject some continuity**

![Stage 1 Overlap Patch Embedding Channels](images/presentation/overlap_patch_embeddings_output.png)
- Initial stage outputs indicate low-level features like edge detection and more abstract features in later stages


### Efficient self-attention
- In the original Transformer paper, authors proposed Query, Key and Values using same dimension. That is referred to as Scaled Dot Product Attention (scaled due to normalizatio applied)
- *But Dot Product Attention operation is O(N<sup>2</sup>)), hence very costly for Images of higher resolution**
- In Efficient Self-Attention, we use a <u>sequence reduction process</u> to reduce the spatial dimension of the key and value by factor R, hence complexity reduces to **O(N<sup>2</sup> / <sub>R</sub>)**

![efficient_self_attention_block_diagram](images/presentation/efficient_self_attention_block_diagram.png)


### Attention Intuition in images
- In segmentation, we're trying to build classes together. So <u>attention will try to group similar classes together, car with car and tree with tree </u>. If depth estimation, pixels containing relevant features will have more weights.
- Attention helps identify related context at location. If you’re looking for car, what is related parts in the entire images
- `Query and key are two different spaces (key is more abstract feature map in efficient version), how to group low level features to get more meaningful vut abstract feature. Attention serves as bridge`. 
![Attention_mechanism](images/presentation/Attention_mechanism.png)


- Simplistically, we can understand attention for 1st pixel as follows : how similar is every pixel in Query to the 1st pixel in Key. This would give attention map for the 1st pixel. This is computed by calculating the Dot Product b/w and Query, softmax. 
![Attention_formula](images/presentation/Attention_formula.PNG)
- Similarly iterating over each pixel in key we'd get attention maps for each.

- [Notebook](finalContent/Efficient_self_attention_intuition.ipynb) explains Efficient Self Attention step by step in images context


### Mix FFN
- Mix Feedforward Network is used as proxy to Positional Encoding (PE) used in Vision Transformers. 
- Authors propose that using PE causes model to perform poor with different test resolution, they use combination of Convolution and Feedforward modules (hence called Mix FFN) to let model learn position information

![Mix_feedforward](images/presentation/Mix_feedforward.png)


### Encoder
- Combining the following modules we get the Transformer block:
    - Efficient Self Attention + LayerNorm 
    - Mix FeedForward + LayerNorm
    - Residual connections

![transformer_block](images/presentation/transformer_block.png)

- Each stage in Encoder consists of one Overlap patch embedding followed by number of Transformer blocks

![mix_encoder_stages](images/presentation/mix_encoder_stages.png)


### All MLP Decoder

- **USP of Segformer compared to previous Transformer models is it combines output of 4 stages of encoder of different resolution using all MLP Decoder**

![Segformer_decoder_arch](images/presentation/Segformer_decoder_arch.png)

- Using Multi-resolution outputs of encoder with simple MLP Decoder helps the model tackle the Multi-scale objects and Global contextual learning.


## HyperParameters
Segformer MiT B3 model was trained with following settings:
- Backbones initialized with imagenet weights and no layers were frozen during training. Decoder weights initialized with kaiming initialization policy
- Plain Cross entropy loss function
- Adam optimizer with OneCycle LR Scheduler with initial learning rate = 3e-4, cosine annealing with 30% ascend
- Batch size of 8 
- 12 epochs of training where model with lowest Dice loss was taken as best model

## Results
- The Segformer MiTB3 model achieves **meanIoU of 70.11%** in testset
- [Notebook](finalContent/Segformer_mit_b3_cityscapes.ipynb) shows end to end training of the model and visualizing results
[![Segformer MiT B3 Semantic Segmentation](images/Segformer_MiT_B3_Cityscapes_semantic_segmentation.gif)](https://youtu.be/NH4xPbxaXAY "Semantic Segmentation Cityscapes using Segformer MiT B3")


## Visualizing Attention Mask
- [Notebook](finalContent/Segformer_visualize_attention.ipynb) can be used to visualise attention masks of different stages of Model as follows:
[![Segformer MiT B3 Attention Head Visualize](images/Segformer_MiT_B3_Cityscapes_Attention_Head_visualize.gif)](https://www.youtube.com/watch?v=BG8MoGAYMkA "Segformer-MiT-B3 Attention heads visualization on Cityscapes dataset")
