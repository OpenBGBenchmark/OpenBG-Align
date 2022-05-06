# OpenBG-Align
"CCKS2022 面向数字商务的知识图谱评测任务二：基于知识图谱的商品同款挖掘"基线方法

# 说明
该仓库主要提供了基于预训练多模态模型CAPTURE进行商品多模态表征抽取，并进行同款挖掘的方法
Capture论文名：《'Product1M: Towards Weakly Supervised Instance-Level Product Retrieval via Cross-modal Pretraining'》
论文链接: https://arxiv.org/abs/2107.14572
相关github仓库：https://github.com/zhanxlin/Product1M

# 使用

请先下载FastRCNN模型[faster_rcnn_from_caffe_attr.pkl](https://drive.google.com/file/d/1NxQumuFWULtnWRGo4p6LtQ1YB2mzdwCu/view?usp=sharing)放到Capture_open/bp_feature文件夹下，下载Capture模型[pytorch_model_8.bin](https://drive.google.com/file/d/1DtYiSQ1fPP1aBGsmIKjz88w1bUcwaTYM/view?usp=sharing)放到Capture_open/Capture文件夹下。

Capture商品多模态表征提取主要分为三个步骤：step0:预训练 step1.基于detectron2对商品图片进行主体特征抽取 step2.综合商品主图+标题进行商品表征抽取
## setp0: 预训练
可跳过，先基于提供的pytorch_model_8.bin进行后续商品表征抽取
```shell script
    sh run_pretrain_task.sh
```

## step1: detectron2 (特征提取)
[bottom-up attention with detectron2](https://github.com/airsplay/py-bottom-up-attention)
### 环境安装
detectron2 需要torch=1.4版本，建议conda配置专门环境跑
```shell script
    git clone https://github.com/airsplay/py-bottom-up-attention.git
    cd py-bottom-up-attention
    ## Install python libraries
    pip install -r requirements.txt
    pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    ## Install detectron2
    python setup.py build develop
   
    ## or if you are on macOS
    # MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
    
    # or, as an alternative to `setup.py`, do
    # pip install [--editable] .
```
### 商品主图主体特征抽取
```shell script
python bp_feature/extract_feature_unit.py \
       --input_file '../item_valid_info.jsonl' \ # 验证集商品信息
       --local_image_path '../item_valid_images/item_valid_images' \
       --output_file './testv1/item_valid_image_feature.csv'  \
       --save_model_path './bp_feature/faster_rcnn_from_caffe_attr.pkl'  # 主体检测模型
```

### 特征格式转化
```shell script
python bp_feature/convert_feature_all.py 
```

## step2: Capture 多模态特征抽取
    可参考Capture/run_inference.ipynb流程
```shell script
    cd Capture
    pip install -r requirements.txt
    sh run_inference.sh
```

# 结果提交
示例代码见Capture/run_inference.ipynb
