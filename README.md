# G-SimCLR : Self-Supervised Contrastive Learning with Guided Projection via Pseudo Labelling

This code provides a TensorFlow implementation and pretrained models for **G-SimCLR** (Guided-SimCLR), as described in the paper [G-SimCLR : Self-Supervised Contrastive Learningwith Guided Projection via Pseudo Labelling]() by Souradip Chakraborty, Aritra Roy Gosthipaty and Sayak Paul.

## Abstract:

In the realms of computer vision, deep neural networks have made their mark. It is noticed that deep models often perform better in a supervised setting with a large amount of labeled data. The representations learned with supervision are not only of high quality but also helps the model in enhancing its accuracy. However, the collection and annotation of a large dataset is extremely expensive and time-consuming. To avoid the same there has been a lot of research going on in the field of unsupervised visual representation learning. 

Among the unsupervised (without labels) training approaches, self-supervised learning has stood out. In the recent advances of self-supervision, [SimCLR](https://arxiv.org/pdf/2002.05709.pdf) by Chen et. al. proves that quality representations can indeed be learned without explicit supervision. In SimCLR, the authors maximize the similarity of augmentations of the same image, and minimize the similarity of augmentations of different images. A linear classifier trained with the representations learned using this approach is able to yield a **76.5%** top-1 accuracy on the ImageNet ILSVRC-2012 dataset. 

In this work, we propose that, with the normalized temperature-scaled cross-entropy (*NT-Xent*) loss function (as used in SimCLR), it is beneficial to not have images of the same category in the same batch. In an unsupervised setting, the information of images pertaining to the same category is missing. We use the latent space representation of a denoising autoencoder trained on the unlabeled dataset and cluster them with k-means to obtain pseudo labels. With this apriori information we batch images, where no two images from the same category are to be found. We report comparable performance enhancements on the CIFAR10 dataset and a subset of the ImageNet dataset. We refer to our method as **G-SimCLR**.

## Datasets Used:

* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

* [ImageNet subset](https://github.com/thunderInfy/imagenet-5-categories)

## Architectures used:

1. [ResNet20](https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer) used for CIFAR10.
2. [ResNet50](https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer) used for ImageNet subset.
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

![Loss Curve](https://github.com/ariG23498/SimCLR_PseudoLabel/blob/master/Assets/Images/Loss_Curves.png)

## Pretrained Weights:

To be updated 

## Results Reported:

### Linear Evaluation

|                                 |      | CIFAR 10  | Imagenet Subset |
| :-----------------------------: | ---- | --------- | --------------- |
|        Fully supervised         |      | 73.62     | 67.6            |
| SimCLR with minor modifications | P1   | 37.69     | 52.8            |
|                                 | P2   | 39.4      | 48.4            |
|                                 | P3   | 39.92     | 52.4            |
|         G-SimCLR (ours)         | P1   | **38.15** | **56.4**        |
|                                 | P2   | **41.01** | **56.8**        |
|                                 | P3   | **40.5**  | **60.0**        |

### Fine-tuning (10% labeled data)

|                                 | CIFAR 10 | Imagenet Subset |
| ------------------------------- | -------- | --------------- |
| Fully supervised                | 73.62    | 67.6            |
| SimCLR with minor modifications | 42.21    | 49.2            |
| G-SimCLR (ours)                 | **43.1** | **56.0**        |







