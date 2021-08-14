# Text Recognition on Cross Domain Datasets
Improved CRNN,ASTER,DAN on different text domains like scene text, hand written, document, chinese/english, even ancient books

# ATTENTION 😮❗😮❗😮❗
This is an experimental project, and the framework is changed every time i uploaded.(Sorry for my mess. I should progame this in a more structured way). Any way, you have to redownload this project every time I uploaded it. So that you can successfully run the code. Thanks for your Attention.🙇‍♂️🙇‍♂️
****
# Update🙂🙂
|Date|Description|
|----|----|
|7/30|Checkpoint for CRNN on IAM dataset has been released. You can test your English handwritten now|
|7/31|Checkpoint for CRNN on CASIA-HWDB2.x has been released. You can test your Chinese handwritten now|
|8/3|New Algorithms! ASTER is reimplemented here and checkpoint for scene text recognition is released|
|8/5|Checkpoint for ASTER on IAM dataset has beem released. It's much more accurate than CRNN due to attention model's implicit semantic information. You should not miss it😃|
|8/8|New Algorithms! DAN(Decoupled attention network) is reimplented. checkpoint forb both scene text and iam dataset are realesed|
|8/11|New Algorithms! ACE(Aggratation Cross-Entropy). It's a new loss function to handle text recognition task. Like CTC and Attention|
****
# 1. Welcome!😃😃
Now I'm focusing on a project to build a general ocr systems which can recognize different text domains. From scene text, hand written, document, chinese, english to even ancient books like confucian classics. So far I don't have a clear idea about how to do it, but let's just do it step by step. This repository is suitable for greens who are interesed in text recognition(I am a green too😂).
****
# 2. Contents👨‍💻👨‍💻
|Part|Description|
|----|----|
|Datasets|[Multible datasets in lmdb form](#datasets)|
|Alogrithms|[CRNN](#71-crnn)|
||[ASTER](#72-aster)|
||[DAN](#43-dan)|
||[ACE]()|
|How to use|[Use](#how-to-use)|
|Checkpoints|[CheckPoints](#checkpoints)|
****
# Datasets
## 3.1 Scene Text Recognitons
### 3.1.1 Training Sets(Synthetic)
|Dataset|Description|Examples|BaiduNetdisk link|
|----|----|----|----|
|SynthText|**9 million** synthetic text instance images from a set of 90k common English words. Words are rendered onto nartural images with random transformations|![SynthText](./github_images/SynthText.JPG)|[Scene text datasets(提取码:emco)](https://pan.baidu.com/s/1PBJf-BtFa7mLkltIfTXPhQ)|
|MJSynth|**6 million** synthetic text instances. It's a generation of SynthText.|![MJText](./github_images/MJSynth.JPG)|[Scene text datasets(提取码:emco)](https://pan.baidu.com/s/1PBJf-BtFa7mLkltIfTXPhQ)|
****
### 3.1.2 Evaluation Sets(Real, and only provide test set)
|Dataset|Description|Examples|BaiduNetdisk link|
|----|----|----|----|
|IIIT5k-Words(IIIT5K)|**3000** test images instances. Take from street scenes and from originally-digital images|![IIIT5K](./github_images/IIIT5K.JPG)|[Scene text datasets(提取码:emco)](https://pan.baidu.com/s/1PBJf-BtFa7mLkltIfTXPhQ)|
|Street View Text(SVT)|**647** test images instances. Some images are severely corrupted by noise, blur, and low resolution|![SVT](./github_images/SVT.JPG)|[Scene text datasets(提取码:emco)](https://pan.baidu.com/s/1PBJf-BtFa7mLkltIfTXPhQ)|
|StreetViewText-Perspective(SVT-P)|**639** test images instances.  It is specifically designed to evaluate perspective distorted textrecognition. It is built based on the original SVT dataset by selecting the images at the sameaddress on Google Street View but with different view angles. Therefore, most text instancesare heavily distorted by the non-frontal view angle.|![SVTP](./github_images/SVTP.JPG)|[Scene text datasets(提取码:emco)](https://pan.baidu.com/s/1PBJf-BtFa7mLkltIfTXPhQ)|
|ICDAR 2003(IC03)|**867** test image instances|![IC03](./github_images/IC03.JPG)|[Scene text datasets(提取码:mfir)](https://pan.baidu.com/s/1PBJf-BtFa7mLkltIfTXPhQ)|
|ICDAR 2013(IC13)|**1015** test images instances|![IC13](./github_images/IC13.JPG)|[Scene text datasets(提取码:emco)](https://pan.baidu.com/s/1PBJf-BtFa7mLkltIfTXPhQ)|
|ICDAR 2015(IC15)|**2077** test images instances. As text images were taken by Google Glasses without ensuringthe image quality, most of the text is very small, blurred, and multi-oriented|![IC15](./github_images/IC15.JPG)|[Scene text datasets(提取码:emco)](https://pan.baidu.com/s/1PBJf-BtFa7mLkltIfTXPhQ)|
|CUTE80(CUTE)|**288** It focuses on curved text recognition. Most images in CUTE have acomplex background, perspective distortion, and poor resolution|![CUTE](./github_images/CUTE.JPG)|[Scene text datasets(提取码:emco)](https://pan.baidu.com/s/1PBJf-BtFa7mLkltIfTXPhQ)|
****
## 3.2 Hand Written
|Dataset|Description|Examples|BaiduNetdisk link|
|----|----|----|----|
|IAM|IAM dataset is based on **handwritten English** text copied from the LOB corpus. It contains 747 documents(**6,482 lines**) in the training set, 116 documents (**976 lines**)in the validation set and 336 documents (2,915 lines) in the testing set|![IAM](./github_images/IAM.JPG)|[IAM_line_level(提取码:u2a3)](https://pan.baidu.com/s/1JqKWHebquezhxtdO8z4Q1Q)|
|CASIA-HWDB2.x|CASIA-HWDB is a large-scale **Chinese hand-written** database.|![CASIA](./github_images/CASIA.JPG)|[HWDB2.x(提取码:ozqu)](https://pan.baidu.com/s/1X-uhmR1i9mWXOGQ9LGjJVA)|

****
# Algorithms
## 4.1 CRNN
### 4.1.1 On Scene Text
- I reimplemented the most classic and wildly deployed algorithm CRNN. The orignal backbone is replaced by a modifyied ResNet and the results below are trained on MJ + ST.

|#|IIIT5K|SVT|IC03|IC13|IC15|SVTP|CUTE|
|----|----|----|----|----|----|----|----|
|CRNN(reimplemented)|**91.2**|**84.4**|**90.8**|**88.0**|**73.1**|**71.8**|**77.4**|
|CRNN(original)|78.2|80.8|89.4|86.7|-|-|-|

- Some recognion results

|Image|GT|Prediction|
|----|----|----|
|![1](./github_images/1.jpg)|I am so sorry|'iamsosory'|
|![2](./github_images/2.jpg)|I still love you|'istilloveyou'|
|![3](./github_images/3.jpg)|Can we begin again|'canwebeginagain'|

- note that we only predict 0-9, a-z. No upper case and punctuations. If you want to predict them, you can modify the code
****
### 4.1.2 On Handwritten
- Relative experiments are conducted on IAM dataset and CASIA-HWDB

|Dataset|Word Accuracy|
|----|----|
|IAM(line level)|67.2|
|CASIA-HWDB2.0-2.2|88.6|

- Some recognion results

|Image|GT|Prediction|
|----|----|----|
|![1](./github_images/IAM2.jpg)|Just Somebody I Can Kiss|'Just Somebody I can kiss'|
|![2](./github_images/IAM3.jpg)|Just something I can turn to|'Just something I can turn to'|
|![3](./github_images/c3.jpg)|昨夜西风凋碧树，独上西楼，望尽天涯路。|'昨夜西风调瑟树,独上西楼。望尽天涯路'|
|![4](./github_images/c4.jpg)|衣带渐宽终不悔，为伊消得人憔悴|'衣带渐宽终不海,为伸消得人憔悴'|
|![5](./github_images/c5.jpg)|众里寻他千百度，蓦然回首，那人却在灯火阑珊处|'众里寻他千百度,暮然回首,那人却在灯火闻班然'|
|![6](./github_images/c6.jpg)|你好，中国|'你好，中国'|
|![7](./github_images/c7.jpg)|欢迎来到重庆|'欢迎来到重庆'|
|![8](./github_images/c8.jpg)|这里是中国，该滚的是你们吧|'这里是中国,该派的是你们吧'|
- Chinese handwritten are sufferd from imbalanced words contribution. So sometimes it's hard to recognize some rare words 
****
## 4.2 ASTER
### 4.2.1 On Scene Text
- ASTER is a classic text recognition algorithms with a **TPS rectification network** and **attention decoder**.

|#|IIIT5K|SVT|IC03|IC13|IC15|SVTP|CUTE|
|----|----|----|----|----|----|----|----|
|ASTER(reimplemented)|**92.9**|88.1|91.2|88.6|75.9|**78.3**|**78.5**|
|ASTER(original)|91.93|**88.76**|**93.49**|**89.75**|#|74.11|73.26|

- Some recognion results

|Image and Rectified Image|GT|Prediction|
|----|----|----|
|![1](./github_images/ASTER_scene1.png)|COLLEGE|'COLLEGE'|
|![2](./github_images/ASTER_scene2.png)|FOOTBALL|'FOOTBALL'|
|![3](./github_images/ASTER_scene3.png)|BURTON|'BURTON'|

****
### 4.2.2 On Handwritten
- Relative experiments are conducted on IAM dataset and CASIA-HWDB

|Dataset|Word Accuracy|
|----|----|
|IAM(line level)|69.8|
|CASIA-HWDB2.0-2.2|The model fails to convergence and I am still training|

- Some recognion results

|Image|GT|Prediction|
|----|----|----|
|![1](./github_images/iam1.jpg)|Coldplay is my favorate band|'Coldplay is my favorate band'|
|![2](./github_images/iam3.jpg)|Night gathers and now my watch begins|'Night gathers and now my watch begins'|
|![3](./github_images/iam4.jpg)|You konw nothing John Snow|'You konw nothing John snow'|
****
****
## DAN
### 4.3.1 On Scene Text

|#|IIIT5K|SVT|IC03|IC13|IC15|SVTP|CUTE|
|----|----|----|----|----|----|----|----|
|DAN1D(reimplemented)|90.6|83.3|88.2|87.5|71.3|71.8|72.6|
|DAN1D(original)|**93.3**|**88.4**|**95.2**|**94.2**|**71.8**|**76.8**|**80.6**|

### 4.3.2 On Handwritten
- Relative experiments are conducted on IAM dataset and CASIA-HWDB

|Dataset|Word Accuracy|
|----|----|
|IAM(line level)|74.0|
|CASIA-HWDB2.0-2.2||

- Some recognion results

|Image|Prediction|
|----|----|
|![1](./github_images/ta1.jpg)|'I have seen things you people would not believe lift'|
|![2](./github_images/ta2.jpg)|'Attack ships on fire off the shoulder of Orien'|
|![3](./github_images/ta3.jpg)|'I have watch  bearans gitter in the does near the Tarhouser'|
|![4](./github_images/ta4.jpg)|'All those moments will be lost in time'|
|![5](./github_images/ta5.jpg)|'like tears in the rain'|
****
## ACE
### 4.4.1 On Scene Text
- ACE is simple yet effective loss funciton. However, there is still a huge gap with CTC and Attention 
  
|#|IIIT5K|SVT|IC03|IC13|IC15|SVTP|CUTE|
|----|----|----|----|----|----|----|----|
|ACE(reimplemented)|78.5|75.6|84.7|78.8|61.5|59.7|63.2|
|ACE(original)|**91.93**|**82.3**|**82.6**|**92.1**|**89.7**|#|#|#|


# How to use
- It's easy to start the training process. Firstly you need to download the datasets
 required.
- Check the root
```
  scripts--
     ACE--
        CASIA_HWDB--
            train.sh
            test.sh
            inferrence.sh 
        iam_dataset--
            train.sh
            test.sh
            inferrence.sh
        scene_text--
            train.sh
            test.sh
            inferrence.sh
     ASTER--
        ...
     CRNN --
        ...
     DAB  --
        ...
```
- let's say you want to train ACE on Scene text. Change the training and testing dataset path in `scripts/ACE/scene_text/train.sh`(The first two rows).
- run
```Bash
bash scripts/ACE/scene_text/train.sh
```
- If you want to test the accuracy, follow the same step as training. Also, you need to set up the resume parameter in .sh. It's where the checkpoint is 
- run
```Bash
bash scripts/ACE/scene_text/test.sh
```
- To test a single image. Change the image path in corresponding .sh and the resume path
- then run
```Bash
bash scripts/ACE/scene_text/inferrence.sh
```
# CheckPoints
## CRNN
### CRNN on Scene Text
[CRNN on STR, Checkpoints(提取码:axf7)](https://pan.baidu.com/s/1Ik6d9aiN8HFCe57pha0Gsg)

### CRNN on IAM dataset
[CRNN on IAM, Checkpoints(提取码:3ajw)](https://pan.baidu.com/s/1_XUzvqgDy4HtRv2F6N34og)

### CRNN on CASIA_HWDB dataset
 [CRNN on CASIA_HWDB, Checkpoints(提取码:ujpy)](https://pan.baidu.com/s/1AfWdvW9ShS09BIiBTIpa4Q)

## ASTER
### ASTER on Scene Text
[ASTER on STR, Checkpoints(提取码:mcc9)](https://pan.baidu.com/s/1jMfLwRJrcfk7IQ5_NDw3-g)

### ASTER on IAM dataset
[ASTER on IAM, Checkpoints(提取码:mqqm)](https://pan.baidu.com/s/1CwxJFKDziZu1dlJCe1gHPg)

##  DAN
### DAN on Scene Text
### DAN on IAM dataset
[DAN on IAM, Checkpoints(提取码:h7vp)](https://pan.baidu.com/s/1lxc3R31AKLyZ_xf_ltiFZA)

# Email 📫
mountchicken@outlook.com

