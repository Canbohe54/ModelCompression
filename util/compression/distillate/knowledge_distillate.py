import torch
import torch.nn as nn
import time


def knowledge_distillate(train_loader, test_loader, teacher_model, student_model, optimizer, device=torch.device('cpu'),
               epochs=10, temperature=7, hard_loss=nn.CrossEntropyLoss(),
               soft_loss=nn.KLDivLoss(reduction="batchmean"), alpha=0.3):
    '''
    知识蒸馏
    :param train_loader: 训练集
    :param test_loader: 测试集
    :param teacher_model: 教师模型
    :param student_model: 学生模型
    :param optimizer: 学生模型优化器
    :param epochs: 迭代次数
    :param device: 训练设备
    :param temperature: 蒸馏温度
    :param hard_loss:
    :param soft_loss:
    :param alpha:
    :return: 最佳模型，最佳代，最佳准确率
    '''
    print("Distillate".center(32, '.'))
    # 记录最好的epoch和acc
    best_acc = 0
    best_epoch = 0
    best_model = student_model
    # 蒸馏方法
    for epoch in range(epochs):
        start_time = time.time()
        distill_loss = 0
        distill_acc = 0
        distill_data_size = 0
        test_loss = 0
        test_acc = 0
        test_data_size = 0
        # 训练集上训练模型权重
        # tqdm_train_loader = tqdm(train_loader)
        # tqdm_train_loader.set_description("epoch %d train process: " %epoch)
        for i, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            # 教师模型预测
            with torch.no_grad():
                teacher_preds = teacher_model(data)

            # 学生模型预测
            student_preds = student_model(data)

            # 计算hard_loss
            student_loss = hard_loss(student_preds, targets)

            # 计算蒸馏后的预测结果及soft_loss
            distillation_loss = soft_loss(
                nn.functional.softmax(student_preds / temperature, dim=1),
                nn.functional.softmax(teacher_preds / temperature, dim=1)
            )

            # 将hard_loss和soft_loss加权求和
            loss = alpha * student_loss + (1 - alpha) * distillation_loss * (temperature * temperature)
            # 乘上T^2平衡数量级

            # 反向传播给学生网络，优化权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stu_pred = torch.argmax(student_preds, dim=1)
            stu_acc = torch.sum(stu_pred == targets)
            distill_loss += distillation_loss.item()
            distill_acc += stu_acc.item()
            distill_data_size += targets.size(0)

        # 测试集上评估模型性能
        student_model.eval()
        num_correct = 0
        num_samples = 0

        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(device)
                targets = targets.to(device)

                preds = student_model(data)

                loss = hard_loss(preds, targets)

                predictions = preds.max(1).indices
                test_acc += torch.sum(predictions == targets)
                test_loss += loss.item()
                test_data_size += targets.size(0)

        student_model.train()
        distill_loss_epoch = distill_loss / distill_data_size
        distill_acc_epoch = distill_acc / distill_data_size
        test_loss_epoch = test_loss / test_data_size
        test_acc_epoch = test_acc / test_data_size
        end_time = time.time()
        memory = torch.cuda.memory_allocated(device=device)
        max_memory = torch.cuda.max_memory_allocated(device=device)
        print(
            'epoch:{} | time:{:.4f}s | mem:{:.4f}MB | max_mem:{:.4f}MB | distill_loss:{:.4f} | distill_acc:{:.4f} | test_loss:{:.4f} | test_acc:{:.4f}'.format(
                epoch + 1,
                end_time - start_time,
                memory / 1048576,
                max_memory / 1048576,
                distill_loss_epoch,
                distill_acc_epoch,
                test_loss_epoch,
                test_acc_epoch))

        # 记录准确率最高
        if test_acc_epoch >= best_acc:
            best_acc = test_acc_epoch
            best_epoch = epoch + 1
            best_model = student_model
        print('Best Accuracy for Validation :{:.4f} at epoch {:d}'.format(best_acc, best_epoch))
    return best_model, best_epoch, best_acc
