# THUHITSZ_Coop-MMAC
THUHITSZ_Coop Team for MICCAI MMAC 2023 - Myopic Maculopathy Analysis Challenge (近视性黄斑病变分析挑战)


[数据集下载](https://pan.baidu.com/s/1hi7ETmqYcJAd7w1oSZEyWA?pwd=lxss#list/path=%2F)

## Task 1 分类任务

Validation Phase - Task 1 | Test Phase - Task 1 | Competition Ends
------------------------- | --------------------| ----------------
June 1, 2023, 8 a.m. UTC | July 15, 2023, 8 a.m. UTC | Aug. 15, 2023, 11:59 p.m. UTC

1. 将数据根据label划分到每个类别的文件夹，并上传至服务器中
2. 先用几个常见的效果很好的分类网络，跑一下结果，看看(CNN:ResNet50, ConvNext-tiny; Transformer:ViT-small, SwinTransformer-Tiny) --- 利用[Timm包](https://github.com/huggingface/pytorch-image-models)
3. 把这些结果都上传上去，先占个坑，同时过一下上传的流程
4. 查看一些相关文献(基于近视性黄斑病变的深度学习文章)，分析现有的做法，然后在此基础上改进

## Task 2 分割

数据集暂时没有公布，继续等待官网

## Task 3 检测

1. 根据label将数据划分到每个文件夹，并上传至服务器
2. 先用yolo、Mask-RCNN两个检测模型跑一下结果，看看
3. 结果上传，熟悉流程
4. 查看相关文章(三种task基本方式一致)


时间规划 | 需要完成的任务 | 是否完成
-------- | -------------| --------
6.5-6.11 | 教大家装深度学习的环境(Pytorch)，并完成数据集的文件夹划分 | False
