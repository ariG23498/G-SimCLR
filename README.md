# G-SimCLR: Self-Supervised Contrastive Learning with Guided Projection via Pseudo Labelling

Official TensorFlow implementation of **G-SimCLR** (Guided-SimCLR), as described in the paper [G-SimCLR: Self-Supervised Contrastive Learning with Guided Projection via Pseudo Labelling](https://arxiv.org/abs/2009.12007) by Souradip Chakraborty<sup>\*</sup>, Aritra Roy Gosthipaty<sup>\*</sup> and Sayak Paul<sup>\*</sup>.

<sup>\*</sup>Equal contribution.

<div align="center"><img src="https://github.com/ariG23498/SimCLR_PseudoLabel/blob/master/Assets/Images/Methodology.png" width="650"></img></div>

*The paper is accepted at [ICDM 2020](http://icdm2020.bigke.org/) for the [Deep Learning for Knowledge Transfer (DLKT)](https://fuzhenzhuang.github.io/DLKT2020/index.html) workshop.* A presentation deck is available [here](https://github.com/ariG23498/G-SimCLR/blob/master/Assets/Deck/G-SimCLR.pdf). 

## Abstract:

In the realms of computer vision,  it is evident that deep neural networks perform better in a supervised setting with a large amount of labeled data. The representations learned with supervision are not only of high quality but also helps the model in enhancing its accuracy. However, the collection and annotation of a large dataset are costly and time-consuming. To avoid the same, there has been a lot of research going on in the field of unsupervised visual representation learning especially in a self-supervised setting. Amongst the recent advancements in self-supervised methods for visual recognition, in SimCLR Chen et al. shows that good quality representations can indeed be learned without explicit supervision. In SimCLR, the authors maximize the similarity of augmentations of the same image and minimize the similarity of augmentations of different images. A linear classifier trained with the representations learned using this approach yields 76.5% top-1 accuracy on the ImageNet ILSVRC-2012 dataset. In this work, we propose that, with the normalized temperature-scaled cross-entropy (NT-Xent) loss function (as used in SimCLR), it is beneficial to not have images of the same category in the same batch. In an unsupervised setting, the information of images pertaining to the same category is missing. We use the latent space representation of a denoising autoencoder trained on the unlabeled dataset and cluster them with k-means to obtain pseudo labels. With this apriori information we batch images, where no two images from the same category are to be found. We report comparable performance enhancements on the CIFAR10 dataset and a subset of the ImageNet dataset. We refer to our method as **G-SimCLR**.     

## Datasets Used:

* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

* [ImageNet subset](https://github.com/thunderInfy/imagenet-5-categories)

## Architectures used:

1. [ResNet20](https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer/blob/master/zoo/resnet/resnet_cifar10.py) used for CIFAR10.
2. [ResNet50](https://keras.io/api/applications/resnet/) used for ImageNet subset.
3. Denoising Autoencoder built from scratch.

## Folder Structure:

```bash
.
├── CIFAR10
│   ├── Autoencoder.ipynb
│   ├── SimCLR_Pseudo_Labels
│   │   ├── Fine_Tune_10_Perc.ipynb
│   │   ├── Linear_Evaluation.ipynb
│   │   └── SimCLR_Pseudo_Labels_Training.ipynb
│   ├── Supervised_Training_CIFAR10.ipynb
│   └── Vanilla_SimCLR
│       ├── Fine_tune_10Perc.ipynb
│       ├── Linear_Evaluation.ipynb
│       └── SimCLR_Training.ipynb
├── Imagenet_Subset
│   ├── Autoencoder
│   │   ├── Deep_Autoencoder.ipynb
│   │   └── Shallow_Autoencoder.ipynb
│   ├── SimCLR_Pseudo_Labels
│   │   ├── Deep Autoencoder
│   │   │   ├── Fine_Tune_10Perc.ipynb
│   │   │   ├── Linear_Evaluation.ipynb
│   │   │   └── SimCLR_Pseudo_Labels_Training.ipynb
│   │   └── Shallow Autoencoder
│   │       ├── Fine_tune_10Perc.ipynb
│   │       ├── Linear_Evaluation.ipynb
│   │       └── SimCLR_Pseudo_Labels_Training.ipynb
│   ├── Supervised_Training_Imagenet_Subset.ipynb
│   └── Vanilla_SimCLR
│       ├── Fine_tune_10Perc.ipynb
│       ├── Linear_Evaluation.ipynb
│       └── SimCLR_Training.ipynb
└── README.md
```

## Loss Curves:

Loss (*NT-Xent*) curves as obtained from the G-SimCLR training with the CIFAR10 and ImageNet Subset datasets respectively.

<div align="center"><img src="https://github.com/ariG23498/SimCLR_PseudoLabel/blob/master/Assets/Images/Loss_Curves.png"></img></div>

## Pretrained Weights:

* [Model Weights (trained on ImageNet Subset)](https://github.com/ariG23498/G-SimCLR/releases/tag/v3.0)
* [Model Weights (trained on CIFAR10)](https://github.com/ariG23498/G-SimCLR/releases/tag/v2.0)

## Results Reported:

### Linear Evaluation

|                                 	|    	| CIFAR 10 	| Imagenet Subset 	|
|:-------------------------------:	|:--:	|:--------:	|:---------------:	|
|         Fully supervised        	|    	|   73.62  	|       67.6      	|
|                                 	| P1 	|   37.69  	|       52.8      	|
| SimCLR with minor modifications 	| P2 	|   39.4   	|       48.4      	|
|                                 	| P3 	|   39.92  	|       52.4      	|
|                                 	| P1 	|   **38.15**  	|       **56.4**      	|
|         G-SimCLR (ours)         	| P2 	|   **41.01**  	|       **56.8**      	|
|                                 	| P3 	|   **40.5**   	|        **60**       	|

### Fine-tuning (10% labeled data)

|                                 	| CIFAR 10 	| Imagenet Subset 	|
|---------------------------------	|:--------:	|:---------------:	|
|         Fully supervised        	|   73.62  	|       67.6      	|
| SimCLR with minor modifications 	|   42.21  	|       49.2      	|
|         G-SimCLR (ours)         	|   **43.1**   	|        **56**       	|

where, 

- **P1** denotes the feature backbone network + the entire non-linear projection head - its final layer
- **P2** denotes the feature backbone network + the entire non-linear projection head - its final two layers
- **P3** denotes the feature backbone network only
