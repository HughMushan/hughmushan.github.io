---
layout: post
title: Deep Learning
---

# Deep Learning 学习笔记（1）——Introduction

## **1. Deep Learning 概述**
### 1.1 Deep Learning 时间表
(1) 1986：提出BP神经网络，模拟人脑神经系统，能够从大量训练样本中学习统计规律。但是存在问题：训练困难；不充分的计算；使用小的训练集；效果并不理想，收敛到局部最优值；从顶层往下，梯度越来越稀疏，误差校正信号越来越小（Gradient Diffusion）;只能用有标签的数据进行训练。
(2) 1986-2006：提出了许多机器学习的模型和算法，如SVM、Boosting、Decision Tree、KNN等。这些现在称为浅层结构；需要为特定任务构建特定的模型；使用的是人工提取的特征（GMM-HMM、SIFT、LBP、HOG）。
(3) 2006：Geffrey Hinton & Ruslan Salakhutdinov 提出Deep belief net。提出了无监督和逐层初始化（Layer-wised pre-training）方法；更好的建模设计；使用了计算机上新的发展技术（GPU&多核计算机系统）；使用了大规模的数据集。认为多层的人工神经网络具有优异的特征学习能力，学习得到的特征对数据更本质刻画，有利于可视化或者分类；深度神经网络在训练上的难度，可以通过逐层初始化来有效克服。
(4) 从2006年开始，机器学习开始在大数据上进行分析。小数据集：容易过度拟合（Overfitting）简化了模型的复杂性；大数据集:容易造成拟合不足（Underfitting），增加模型的复杂性，需要的计算资源大，优化难度大。
(5) 2011：Microsoft 在语音上使用了Deep Learning技术，并取得了突破性的成果。
(6) 2012：在ImageNet competition上多伦多大学使用了Deep Learning技术，取得第一名。论文：ImageNet classification with Deep Convolutional Neural Networks。
(7) 2013：ImageNet classification challenge Top20全用的是Deep Learning技术。在ImageNet object detection challenge上NYU使用Deep Learning技术取得Rank3成绩。此外Google和百度先后使用Deep Learning技术，并在上面做深入研究。
(8) 2014: ImageNet challenge基本上全用的是Deep Learning。Deep Learning被认为是计算机视觉方面效果最好的方法（State of art）。使用DeepLearning，在LFW数据集上的Face verification准确率可以达到99.47%。论文: Deep Learning Face Representation by joint identification-verification; Deeply learned face representation are spare, selective, and robust.

**图片**

### 1.2 Deep Learning 基本原理
机器学习在2006年出现新的浪潮。在2006年之前的称为是浅层学习(Shallow Learning)，Deep Learning是机器学习的第二次浪潮。之所以称06年之前的机器学习算法或Shallow Learning，是因为它们都可以近似的看作带有一层隐层节点的神经网络。Deep Learning与Shallow Learning的主要区别在于两个方面：(1) 特征选择方式；(2)结构方面。
在特征方面，Shallow Learning的效果很大程度上取决于特征提取的好与坏，而特征往往是人工设计，具有局限性。跟人在领域上的知识相关；特征的设定与训练相互独立；如果人工特征有多个参数，非常困难进行人工微调；特定应用对应特定特征，为一个新的应用找到新的有效特征非常缓慢。Deep Learning更加强调了特征的学习（Feature Learning），
