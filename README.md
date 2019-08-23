# Content-based Image Retrieval (CBIR)
## Dataset
  进入datasets，根据提示下载相关数据集
## Pipeline
  执行下述命令，得到检索的mAP
```bash
python ./CBIR/main.py --dataset_name=Holiday 
```
## 实验记录
- 20190726
```bash
the baseline mAP is 0.7643
resnet_18: mAP = 0.6240
vgg_16(before relu): mAP = 0.5936
vgg_16(after relu): mAP = 0.6563
``` 





