# Mine-Extra-Knowledge
Official implementation of the NeurIPS2023 paper: Leave No Stone Unturned: Mine Extra Knowledge for Imbalanced Facial Expression Recognition



## Abstract
Facial expression data is characterized by a significant imbalance, with most collected data showing happy or neutral expressions and fewer instances of fear or disgust. This imbalance poses challenges to facial expression recognition (FER) models, hindering their ability to fully understand various human emotional states. Existing FER methods typically report overall accuracy on highly imbalanced test sets but exhibit low performance in terms of the mean accuracy across all expression classes. In this paper, our aim is to address the imbalanced FER problem. Existing methods primarily focus on learning knowledge of minor classes solely from minor-class samples. However, we propose a novel approach to extract extra knowledge related to the minor classes from both major and minor class samples. Our motivation stems from the belief that FER resembles a distribution learning task, wherein a sample may contain information about multiple classes. For instance, a sample from the major class surprise might also contain useful features of the minor class fear. Inspired by that, we propose a novel method that leverages re-balanced attention maps to regularize the model, enabling it to extract transformation invariant information about the minor classes from all training samples. Additionally, we introduce re-balanced smooth labels to regulate the cross-entropy loss, guiding the model to pay more attention to the minor classes by utilizing the extra information regarding the label distribution of the imbalanced training data. Extensive experiments on different datasets and backbones show that the two proposed modules work together to regularize the model and achieve state-of-the-art performance under the imbalanced FER task.



## Train



**Dataset**

[RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset) is aligned for Swin-T and should be incorporated under the folder ./dataset

You can download the dataset from this [link](https://drive.google.com/file/d/1GiVsA5sbhc-12brGrKdTIdrKZnXz9vtZ/view?usp=drive_link)

**Pretrained backbone model**

Pretrained model should be included under the code directory and can be downloaded from this [link](https://drive.google.com/file/d/1GiVsA5sbhc-12brGrKdTIdrKZnXz9vtZ/view?usp=drive_link). 

**Train the model**

Just run one line of code to train the model. 

```key
sh train_exp.sh
```


## Results



**Accuracy**

Traing the model on RAF-DB clean train set (Swin-T backbone) should achieve over 92.31\% accuracy on RAF-DB test set. The best accuracy could reach 92.86\%. As for the mean accuracy of all classes, it should reach 87.71\%.

![](https://github.com/zyh-uaiaaaa/Mine-Extra-Knowledge/blob/main/code/imgs/accuracy.png)







