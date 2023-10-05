import torch
import time
import torch.nn as nn


def train(train_loader, test_loader, model, optimizer, epochs=10, loss_func=nn.CrossEntropyLoss(), device=torch.device('cpu')):
    '''
    训练模型
    :param train_loader: 训练集
    :param test_loader: 测试集
    :param model: 模型
    :param loss_func: 损失函数
    :param optimizer: 优化器
    :param device: 训练设备
    :param epochs: 迭代次数
    :return:最佳模型，最佳代，最佳准确率
    '''
    print('Train'.center(32,'.'))
    # 记录每个epoch的loss和acc
    best_acc = 0
    best_epoch = 0
    best_model = model
    # 训练过程
    for epoch in range(1, epochs):
        # 设置计时器，计算每个epoch的用时
        start_time = time.time()
        model.train()  # 保证每一个batch都能进入model.train()的模式
        # 记录每个epoch的loss和acc
        train_loss, train_acc, test_loss, test_acc = 0, 0, 0, 0
        train_data_size = 0
        test_data_size = 0
        for i, (inputs, labels) in enumerate(train_loader):
            # print(batch_size)
            # print(i, inputs, labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 预测输出
            outputs = model(inputs)
            # 计算损失
            loss = loss_func(outputs, labels)
            # print(outputs)
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
                # print(pred,'================')
                # print(pred==labels,'=====----------======')
                acc = torch.sum(pred == labels)
                # acc = calculat_acc(outputs, labels)
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
            'epoch:{} | time:{:.4f}s | mem:{:.4f}MB | max_mem:{:.4f}MB | train_loss:{:.4f} | train_acc:{:.4f} | test_loss:{:.4f} | test_acc:{:.4f}'.format(
                epoch,
                end_time - start_time,
                memory / 1048576,
                max_memory / 1048576,
                train_loss_epoch,
                train_acc_epoch,
                test_loss_epoch,
                test_acc_epoch))
        # print(torch.cuda.memory_stats(device=device))

        # 记录验证集上准确率最高的模型
        if test_acc_epoch >= best_acc:
            best_acc = test_acc_epoch
            best_epoch = epoch
            best_model = model
        print('Best Accuracy for Validation :{:.4f} at epoch {:d}'.format(best_acc, best_epoch))
    return best_model, best_epoch, best_acc
