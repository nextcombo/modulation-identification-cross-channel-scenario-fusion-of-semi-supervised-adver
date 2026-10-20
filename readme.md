# [Modulation Identification] Cross-Channel Scenario: Fusion of Semi-Supervised Adversarial Domain Adaptive Modulation Classification Network [with Python Code]

1. Abstract: Autonomous modulation classification (AMC) based on deep learning technology has become a research hotspot and demonstrates numerous practical application values. However, the performance of AMC deep learning networks trained on specific channel models in new channel scenarios is poor. To address this issue, this paper proposes an adversarial semi-supervised domain adaptation method. Specifically, the method optimizes classification accuracy, cluster sensitivity, and distribution distance through a three-stage iterative training process. Finally, the proposed model can achieve robust performance even when only a small amount of labeled data is used when migrating to new channels.
2. Introduction: In the spectrum perception of the sixth generation (6G) network, autonomous modulation classification (AMC) is an important means. Although AMC methods based on deep learning (DL) have achieved significant results with large datasets, they perform poorly in unknown channel environments. To solve this problem, during training, a new dataset containing new transmitter parameters and channel conditions should be configured to ensure the model adapts to the new environment. However, obtaining sufficient labeled data in non-cooperative communication scenarios poses great challenges. Therefore, the focus of research has shifted towards how to achieve model migration between different environments and maximize the use of limited labeled data to address this issue.
In recent years, transfer learning (TL) has emerged as a novel method for transferring knowledge across different environments. Perenda et al. introduced transfer learning and spatial transformation networks (STN) into the ResNeXt-based modulation classifier to ensure the model's robustness under unknown shaping parameters (up-sampling factors, pulse shaping filter parameters, and sampling frequency) and channel models (additive Gaussian white noise, Rayleigh, Rician); Zhang et al. focused on enhancing the generalization capability of models in unknown modulation scheme environments through leveraging transfer learning. They proposed an open set domain adaptation AMC (OSDA-AMC) algorithm by combining supervised adversarial domain adaptation techniques with open set modulation recognition (OSMR).Zhou et al., and Liang et al., respectively, conduct knowledge transfer between different AMC datasets (RML2016.10A, RML2016.10C, etc.). The former reuses pre-trained networks on large-scale datasets, while the latter proposes an improved domain adversarial neural network (DANN) that incorporates self-attention mechanisms to preserve specific domain features of both the target and source domains. However, most research related to transfer learning in AMC lacks sufficient exploration of cross-channel scenarios, which significantly differ from the distribution of practical channel environments in practice.
Semi-Supervised Learning (SSL) is another learning paradigm that combines labeled and unlabeled data to train machine learning models, which can alleviate the burden of data labeling. It is suitable for scenarios where labeled data is insufficient. Liu et al. proposed a pre-training method based on self-supervised contrastive learning, utilizing a large amount of unlabeled data for pre-training, followed by fine-tuning on a small amount of labeled data for downstream tasks. On this basis, Bai et al. considered both modality-level and instance-level feature representations during the pre-training process of unlabeled data, extracting more robust features. However, these two methods essentially only train on the target domain, unable to leverage the correlation between different channel environments.Guo et al. propose a Semi-Supervised Autonomous Modulation Recognition (SemiAMR) framework, utilizing teacher-student models and pseudo-labeling training methods to address the issue of insufficient labels. Although semi-supervised learning in the AMC domain has made some progress, most current research assumes that the distribution of training and test datasets is identical, which poses new challenges for research in this field due to the differences in data distribution.
In summary, existing semi-supervised learning based AMC methods assume that the training and test datasets have consistent distributions; while studies based on transfer learning can transfer knowledge between datasets with different distributions, they seldom involve cross-channel scenarios. In this regard, semi-supervised domain adaptation (SSDA) can be considered as a viable solution.
In machine learning, a "domain" refers to the specific knowledge domain or field that a particular machine learning model is applied to. Domain shift refers to the statistical characteristics of data distribution changing during the training and deployment stages of a machine learning model. Different channel environments correspond to different domains, leading to domain shifts. Domain adaptation (DA) can be used to address domain shift issues. It is a machine learning paradigm aimed at solving the challenge of transferring knowledge from a source domain (rich in labeled data and easily accessible) to a target domain (poor in labeled data or non-existent). In cross-channel scenarios, the AMC dataset sampled from known training channel environments can be considered as the source domain; the AMC dataset sampled from actual test channel environments can be seen as the target domain, which lacks sufficient labels.The distribution offset of wireless channels will cause domain offset. Domain adaptation can find a distribution. Enable the modulation schemes of both source and target domains to be effectively classified. Semi-supervised domain adaptation utilizes insufficient labeled data in the target domain. It can be considered as a combination of domain adaptation and semi-supervised learning.
Inspired by semi-supervised domain adaptation research, this paper proposes a novel hybrid semi-supervised adversarial domain adaptation (FSSADA) framework suitable for cross-channel scenarios. This framework integrates labeled and unlabeled data information effectively, with the main contributions as follows:
1.  
A novel three-stage iterative training method is proposed, which effectively integrates labeled and unlabeled data through supervision and self-training.
2.  
A new domain adaptation method is introduced, which constrains the distribution of source and target domains in the feature space. Optimizing cluster sensitivity can enhance the similarity of samples with the same modulation type in the feature space; imposing distribution consistency constraints can better narrow the distribution difference between source and target domain samples in the feature space.
3.  
The proposed method is validated under different channel conditions and AMC datasets, achieving robust results.
III. Network Architecture (I) The integrated hybrid semi-supervised adversarial domain adaptation (FSSADA) framework proposed in this paper consists of three iterative training stages, detailed as follows:
1.  
Supervised domain training (ST) stage: Using the AMC dataset with both source and labeled target domains, combined with labels for training to optimize model performance on known label data.
2.  
Target domain self-training (TDST) stage: Self-train the unlabeled target domain dataset without labels, introducing cluster sensitivity to constrain the distribution of the target domain. Among them, Gradient Reversal Layer (GRL) is a technique in domain adaptation, reversing gradients during backpropagation to make the model learn features that are not sensitive to domain shifts.
3.  
Fusion domain self-training (FDST) stage: Both source and target domain datasets undergo self-training without labels to narrow the distribution difference between them.
The framework's network consists of three parts:
•  
Shared feature extraction module (): Responsible for extracting hidden features from input data, with parameters.
1. Distribution Distance Representation Module (DDR): It receives and outputs simultaneously, computing and outputting the loss, with parameters.

2. Supervised Domain Training (ST):
In this phase, the model's performance is optimized using labeled datasets from both source and target domains, calculating losses for each domain as follows:

where,
- represents a binary indicator variable (0 or 1), indicating whether the category label correctly classifies the sample;
- is the predicted probability of the sample belonging to a certain category given the model parameters and;
- is the total number of labeled samples in both source and target domains;
- is the number of categories.

3. Target Domain Self-Training (TDST):
This phase mainly introduces cluster sensitivity to constrain the distribution of the target domain, with cluster sensitivity loss defined by entropy:

where,
- is the element at position in dimension, representing the estimated probability that a sample belongs to class. If set, only when mapped to a uniform distribution will it reach its maximum value. This reflects the distance from the cluster centers of various classes, where a uniform distribution implies poor cluster sensitivity, making it difficult to distinguish between different classes.

Adversarial training is introduced in this phase, with the objective function being:
By integrating the theory of Generative Adversarial Networks (GANs), the hidden features of labeled source domain data and a small amount of target domain data can be considered as real data, while the hidden features of unlabeled target domain data can be considered as generated data. The goal is to make the generated data approach real data. The feature extraction module can serve as a generator, and the classification module as a discriminator. Therefore, the feature extraction module needs to reduce the cluster sensitivity loss, while the classification module needs to enhance the cluster sensitivity loss to improve discrimination capability, which aligns with the above objective function.

4. Fusion Domain Self-Training (FDST):
The phase combines training on both the source and target domains to reduce the distributional difference of hidden features. The Wasserstein Distance is used to measure the distance between two probability distributions (source domain distribution) and (target domain distribution), defined as:

where, are all possible joint probability distributions between and, respectively, with and being the marginal distributions of and, i.e., and vice versa. This formulation can be understood as the minimum energy cost required to move a "heap" of distributions from one to another.

Since iterating over all possible joint distributions is impractical, the formula is transformed based on the Kantorovich-Rubinstein duality:

where, the real-valued function must satisfy - Lipschitz continuity, meaning there exists a constant, for all, such that, simplifying for this paper by setting to 1.

The distribution distance represents the module calculating the Wasserstein Distance between and using the transformed formula (i.e., the above equation). The distribution distance loss is defined as:

According to the transformed Wasserstein Distance formula, the real-valued function needs to be subjected to - Lipschitz continuity constraints, hence gradient penalty is introduced:

where, located in the hidden feature space of and.
Maximizing the transformed Wasserstein Distance formula ensures a better distance metric to measure the two probability distributions; minimizing the distribution distance loss ensures the closeness of two distributions in the hidden space. The optimization of the distribution distance representation module can be represented as:

where, is the balancing factor, and is the loss. During training, alternating adversarial training is employed to train and.

(III) Key module details 1
Feature extraction module (): ResNet-18 network is utilized as the core part to extract hidden features, with its structure parameters as shown in the table below:

| Level | Output size | Conv kernel size |  
| ---- | ---- | ---- |  
| Input signal | 1x2x1024|-|
| Convolutional Layer 1 (Conv1) | 64x2x256 | 1x4 |
Residual Block 1 (ResBlock1) | 64x2x256|3x3|
Residual Block 2 (ResBlock2) |128x1x128|3x3|
Residual Block 3 (ResBlock3) |256x1x64|3x3|
Residual Block 4 (ResBlock4) |512x1x32|3x3|
| Average Pooling Layer (AvgPool) | 512x1x1 | - |
Flattening Layer | 512 | - |
| Fully Connected Layer (Linear) | 128 | - |
2  
Classification module (: a single-layer classification head responsible for outputting the probability distribution of categories.
3  
Distribution distance: The module (): contains a Gradient Reversal Layer (GRL) and three fully connected layers, with output dimensions of 512, 128, and 1 respectively. During the backpropagation process in adversarial training, the gradient reversal layer reverses the gradients, while it is not enabled during regular training.
IV. Conclusion This paper proposes a semi-supervised domain adaptation framework to address the cross-channel automatic modulation classification (AMC) scenario. By combining distributional distance representation with classification cluster sensitivity, this framework outperforms other comparative methods in the cross-channel setting. However, there are still many directions worth exploring in this field, such as how to achieve rapid adaptation to unknown channels using less data, which will be one of the focuses of future research.
End

## Images

![[Modulation Identification] Cross-channel scenario: Semi-Supervised Adversarial Domain Adaptive Modulation Classification Network [with Python code]](01.jpg)
![[Modulation Identification] Cross-channel scenario: Semi-Supervised Adversarial Domain Adaptive Modulation Classification Network [with Python code]](02.jpg)
![[Modulation Identification] Cross-channel scenario: Semi-Supervised Adversarial Domain Adaptive Modulation Classification Network [with Python code]](03.jpg)
![[Modulation Identification] Cross-channel scenario: Semi-Supervised Adversarial Domain Adaptive Modulation Classification Network [with Python code]](04.jpg)
![[Modulation Identification] Cross-channel scenario: Semi-Supervised Adversarial Domain Adaptive Modulation Classification Network [with Python code]](05.jpg)
![[Modulation Identification] Cross-channel scenario: Semi-Supervised Adversarial Domain Adaptive Modulation Classification Network [with Python code]](06.jpg)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
