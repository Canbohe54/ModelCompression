import torch.nn as nn

class CnnNet_mnist_4_8(nn.Module):
    def __init__(self, classes=10):
        super(CnnNet_mnist_4_8, self).__init__()
        # 分类数（默认有10种数字）
        self.classes = classes
        # 第一层卷积，输入：bs*1*28*28 输出：bs*4*14*14
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            # 归一化
            nn.BatchNorm2d(4),
            # 激活函数
            nn.ReLU(),
            # 最大池化：输入:bs*4*28*28  输出：bs*4*14*14
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # 第二层卷积，输入：bs*4*14*14 输出：bs*8*7*7
        self.conv2 = nn.Sequential(
            # 卷积 输入：bs*4*14*14  输出：bs*8*14*14
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            # 归一化
            nn.BatchNorm2d(8),
            # 激活函数
            nn.ReLU(),
            # 最大池化：输入:bs*8*14*14  输出：bs*8*7*7
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # 自适应池化，将bs*8*3*3映射为bs*8*1*1
        self.advpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.fc = nn.Linear(8, self.classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.advpool(x)
        # 需要将多维度的值展平为一维，送入linear中，但是需要保持batchsize的维度
        # 例如2*64*1*1 变成2*64
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out