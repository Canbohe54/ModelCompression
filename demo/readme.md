# demo
### Requirements

- python 3
- pyyaml
- numpy
- pandas
- pytorch >= 2.0.0
- torchvision
- torchsummary
- thop
- ptflops

### Run the demo
- create 3 folder in demo

```
/data   /origin_model   /output_model
```

- run these commands in turn:

```
    python origin_train.py

    python DAFL.py

    python evaluate.py
```
### Change configs
```yaml
    classes: 10
    #input_model_type: "resnet18"
    #output_model_type: "shufflenet_v2_x0_5"
    input_model_path: "./origin_model/origin.pth"
    output_model_path: "./output_model/student.pth"
    img_size: 32
    img_channels: 3
    
    demo_origin_model_train:
      dataset: "cifar10"
      dataset_path: "./data"
      model_type: "resnet34"
      model_path: "./origin_model/origin_1.pth"
      train_epochs: 2000
      batch_size: 32
      learning_rate: 0.1
    
    DAFL:
      teacher_model_dir: "./origin_model/origin.pth" # 教师模型/原模型位置
      teacher_model_structured: False
      teacher_model_type: "resnet34"
      student_model_dir: "./output_model/student.pth" # 学生模型/压缩后模型位置
      student_model_type: "resnet18"
      distill_epochs: 2000 # 蒸馏次数
      adversaries: 360 # 每个epoch对抗次数
      batch_size: 128
      learning_rate_generator: 0.2 # 生成器学习率
      learning_rate_student: 2e-5 # 学生网络学习率
      latent_dim: 1000 #全连接维度
      img_size: 32 # 图像像素大小
      channels: 3 # 图像通道数
      # 损失函数参数
      one_hot_loss_alpha: 0.05
      information_entropy_loss_beta: 5
      activation_loss_gamma: 0.1
```
### Update logs
- 2023/10/25
  - 在pytorch2复现DAFL方法
    - 加入forward_with_feature方法，支持torchvison.models库模型
    - 更改为外挂config.yaml方式传入参数
  - 移植torchstat到pytorch2，并修正了一些问题
    - 修复torchstat与torchsummary不兼容的问题
    - 修复torchstat与新版numpy不兼容的问题
    - 修复torchstat与新版pandas不兼容的问题
    - 修复MAdd与FLOPs显示错误问题