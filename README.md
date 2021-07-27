# CRNN Text Recognition on Cross Domain Datasets
Improved CRNN on different text domains like scene text, hand written, document, chinese/english, even ancient books

****
# 1. Welcome!😃😃
Now I'm focusing on a project to build a general ocr systems which can recognize different text domains. From scene text, hand written, document, chinese, english to even ancient books like confucian classics. So far I don't have a clear idea about how to do it, but let's just do it step by step. This repository is suitable for greens who are interesed in text recognition(I am a green too😂).
****
# 2. Contents👨‍💻👨‍💻
|Part|Description|
|----|----|
|Datasets|[Multible datasets in lmdb form](#Datasets)|
|Alogrithms|[CRNN](#Algorithms)|
|Train|[Train](#Train)|
|Test|[Test](#Test)|
|Inferrence|[Inferrence](#Inferrence)|
****
# Datasets
## 3.1 Scene Text Recognitons
### 3.1.1 Training Sets(Synthetic)
|Dataset|Description|Examples|BaiduNetdisk link|
|----|----|----|----|
|SynthText|**9 million** synthetic text instance images from a set of 90k common English words. Words are rendered onto nartural images with random transformations|![SynthText](./github_images/SynthText.JPG)|[Scene text datasets(提取码:mfir)](https://pan.baidu.com/s/1iGAO_TcAeNDdrzIbf2pjXw)|
|MJSynth|**6 million** synthetic text instances. It's a generation of SynthText.|![MJText](./github_images/MJSynth.JPG)|[Scene text datasets(提取码:mfir)](https://pan.baidu.com/s/1iGAO_TcAeNDdrzIbf2pjXw)|
****
### 3.1.2 Evaluation Sets(Real, and only provide test set)
|Dataset|Description|Examples|BaiduNetdisk link|
|----|----|----|----|
|IIIT5k-Words(IIIT5K)|**3000** test images instances. Take from street scenes and from originally-digital images|![IIIT5K](./github_images/IIIT5K.JPG)|[Scene text datasets(提取码:mfir)](https://pan.baidu.com/s/1iGAO_TcAeNDdrzIbf2pjXw)|
|Street View Text(SVT)|**647** test images instances. Some images are severely corrupted by noise, blur, and low resolution|![SVT](./github_images/SVT.JPG)|[Scene text datasets(提取码:mfir)](https://pan.baidu.com/s/1iGAO_TcAeNDdrzIbf2pjXw)|
|StreetViewText-Perspective(SVT-P)|**639** test images instances.  It is specifically designed to evaluate perspective distorted textrecognition. It is built based on the original SVT dataset by selecting the images at the sameaddress on Google Street View but with different view angles. Therefore, most text instancesare heavily distorted by the non-frontal view angle.|![SVTP](./github_images/SVTP.JPG)|[Scene text datasets(提取码:mfir)](https://pan.baidu.com/s/1iGAO_TcAeNDdrzIbf2pjXw)|
|ICDAR 2003(IC03)|**867** test image instances|![IC03](./github_images/IC03.JPG)|[Scene text datasets(提取码:mfir)](https://pan.baidu.com/s/1iGAO_TcAeNDdrzIbf2pjXw)|
|ICDAR 2013(IC13)|**1015** test images instances|![IC13](./github_images/IC13.JPG)|[Scene text datasets(提取码:mfir)](https://pan.baidu.com/s/1iGAO_TcAeNDdrzIbf2pjXw)|
|ICDAR 2015(IC15)|**2077** test images instances. As text images were taken by Google Glasses without ensuringthe image quality, most of the text is very small, blurred, and multi-oriented|![IC15](./github_images/IC15.JPG)|[Scene text datasets(提取码:mfir)](https://pan.baidu.com/s/1iGAO_TcAeNDdrzIbf2pjXw)|
|CUTE80(CUTE)|**288** It focuses on curved text recognition. Most images in CUTE have acomplex background, perspective distortion, and poor resolution|![CUTE](./github_images/CUTE.JPG)|[Scene text datasets(提取码:mfir)](https://pan.baidu.com/s/1iGAO_TcAeNDdrzIbf2pjXw)|
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

# Train
## 5.1 Train CRNN
### 5.1.1 Train CRNN on Scene Text
- modify `scripts/train.sh`. Set up training set path and evaluation set path.
- run 
```Bash
bash scripts/train.sh
```
- If you want to predict more characters, add a line in `scripts/train.sh`
 ```Bash
-- alphabets 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ \
or
-- alphabets 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ \
```
# Test
## 6.1 Test CRNN
### 6.1.1 Test CRNN on Scene Text
- DownLoad Pretrained Checkpoints, [CRNN on STR, Checkpoints(提取码:o4mb)](https://pan.baidu.com/s/1aHiU1mDvYD5gs2G9Zr3JjQ)
- modify `scripts/test.sh`. Set up test_data_dir and resume for the checkpoint
- run 
```Bash
bash scripts/test.sh
```

# Inferrence
## 7.1 Inferrence CRNN
- DownLoad Pretrained Checkpoints, [CRNN on STR, Checkpoints(提取码:o4mb)](https://pan.baidu.com/s/1aHiU1mDvYD5gs2G9Zr3JjQ)
- modify `scripts/inferrence.sh`. Set up image_path and resume for the checkpoint
- run 
```Bash
bash scripts/inferrence.sh
```
