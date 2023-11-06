import time

import torch
import yaml
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import datasets
import torch.nn as nn


def adjust_learning_rate(optimizer, epoch, learning_rate):  # lr优化
    if epoch < 80:
        lr = learning_rate
    elif epoch < 120:
        lr = 0.1 * learning_rate
    else:
        lr = 0.01 * learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def readconfig(config_path="config.yaml"):
    global classes, dataset, dataset_path, model_type, model_path, train_epochs, batch_size, learning_rate
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        classes = config["classes"]
        model_train_config = config["demo_origin_model_train"]
        dataset = model_train_config["dataset"]
        dataset_path = model_train_config["dataset_path"]
        model_type = model_train_config["model_type"]
        model_path = model_train_config["model_path"]
        train_epochs = model_train_config["train_epochs"]
        batch_size = model_train_config["batch_size"]
        learning_rate = model_train_config["learning_rate"]


def train(train_loader, test_loader, model, optimizer, learning_rate, epochs=10, loss_func=nn.CrossEntropyLoss(),
          device=torch.device('cpu')):
    print('Train'.center(32, '.'))
    # 记录每个epoch的loss和acc
    best_acc = 0
    best_epoch = 0
    best_model = model
    # 训练过程
    for epoch in range(1, epochs):
        adjust_learning_rate(optimizer=optimizer, epoch=epoch, learning_rate=learning_rate)
        # 设置计时器，计算每个epoch的用时
        start_time = time.time()
        model.train()  # 保证每一个batch都能进入model.train()的模式
        # 记录每个epoch的loss和acc
        train_loss, train_acc, test_loss, test_acc = 0, 0, 0, 0
        train_data_size = 0
        test_data_size = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 预测输出
            outputs = model(inputs)
            # 计算损失
            loss = loss_func(outputs, labels)
            # 因为梯度是累加的，需要清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 优化器
            optimizer.step()
            # 计算准确率
            output = nn.functional.softmax(outputs, dim=1)
            pred = torch.argmax(output, dim=1)
            acc = torch.sum(pred == labels)
            train_loss += loss.item()
            train_acc += acc.item()
            train_data_size += labels.size(0)
        # 验证集进行验证
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 预测输出
                outputs = model(inputs)
                # 计算损失
                loss = loss_func(outputs, labels)
                # 计算准确率
                output = nn.functional.softmax(outputs, dim=1)
                pred = torch.argmax(output, dim=1)
                acc = torch.sum(pred == labels)
                test_loss += loss.item()
                test_acc += acc.item()
                test_data_size += labels.size(0)
        # 计算每个epoch的训练损失和精度
        train_loss_epoch = train_loss / train_data_size
        train_acc_epoch = train_acc / train_data_size
        # 计算每个epoch的验证集损失和精度
        test_loss_epoch = test_loss / test_data_size
        test_acc_epoch = test_acc / test_data_size
        end_time = time.time()
        memory = torch.cuda.memory_allocated(device=device)
        max_memory = torch.cuda.max_memory_allocated(device=device)
        print(
            'epoch:{}/{} | time:{:.4f}s | mem:{:.4f}MB | max_mem:{:.4f}MB | train_loss:{:.4f} | train_acc:{:.4f} | test_loss:{:.4f} | test_acc:{:.4f}'.format(
                epoch,
                epochs,
                end_time - start_time,
                memory / 1048576,
                max_memory / 1048576,
                train_loss_epoch,
                train_acc_epoch,
                test_loss_epoch,
                test_acc_epoch))
        # 记录验证集上准确率最高的模型
        if test_acc_epoch >= best_acc:
            best_acc = test_acc_epoch
            best_epoch = epoch
            best_model = model
            torch.save(best_model, model_path)
        print('Best Accuracy for Validation :{:.4f} at epoch {:d}'.format(best_acc, best_epoch))
    return best_model, best_epoch, best_acc


def main():
    readconfig()
    if dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_data = datasets.CIFAR10(root=dataset_path, train=True, download=False, transform=transform_train)
        test_data = datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=transform_test)
    if dataset == "mnist":
        train_data = datasets.MNIST(root=dataset_path, train=True, download= False, transform=transforms.Compose([
                           transforms.Resize((7, 7)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        test_data = datasets.MNIST(root=dataset_path, train=False, download= False, transform=transforms.Compose([
                           transforms.Resize((7, 7)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                           ]))
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    print('训练数据集大小：', len(train_loader))
    print('测试数据集大小：', len(test_loader))
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    print("图片大小", images.shape)  # 打印图像数据的形状，如 (64, 1, 28, 28)
    print("标签数据大小", labels.shape)  # 打印标签数据的形状，如 (64)

    if model_type == "resnet34":
        model = models.resnet34(num_classes=classes)
    if model_type == "resnet18":
        model = models.resnet18(num_classes=classes)
    if model_type == "mobilenet_v2":
        model = models.mobilenet_v2(num_classes=classes)
    if model_type == "shufflenet_v2_x0_5":
        model = models.shufflenet_v2_x0_5(num_classes=classes)
    if model_type == "alexnet":
        model = models.alexnet(num_classes=classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = torch.load(model_path)
    model.to(device=device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    train(train_loader, test_loader, model, optimizer, learning_rate, train_epochs, loss_func, device)


if __name__ == "__main__":
    main()
