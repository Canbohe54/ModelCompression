import torchvision.models
import yaml
import os



from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as nnfunc
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms


def kd_loss(y, teacher_scores):
    p = nnfunc.log_softmax(y, dim=1)
    q = nnfunc.softmax(teacher_scores, dim=1)
    l_kl = nnfunc.kl_div(p, q, reduction='sum') / y.shape[0]
    return l_kl

def adjust_learning_rate(optimizer, epoch, learing_rate):  # 动态调整学习率
    if epoch < 20:
        lr = learing_rate
    elif epoch < 80:
        lr = 0.1 * learing_rate
    elif epoch < 240:
        lr = 0.01 * learing_rate
    elif epoch < 600:
        lr = 0.001 * learing_rate
    elif epoch < 1200:
        lr = 0.0005 * learing_rate
    else:
        lr = 0.0001 * learing_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def forward_with_feature_output(model, input): # 重写forward，带feature输出
    # 适用大部分模型结构
    mo = nn.Sequential(*list(model.children())[:-1])
    feature = mo(input)
    # feature = nnfunc.avg_pool2d(feature, 1)
    feature = feature.view(input.size(0), -1)
    output = list(model.modules())[-1](feature)
    return output, feature
class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, channels):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(channels, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img

class DAFL:
    def __init__(self,config_path=""):
        # 训练设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 配置文件
        self.classes = 10
        self.teacher_model_dir = "./origin_model/teacher.pth"
        self.teacher_model_structured = False
        self.teacher_model_type = "resnet50"
        self.student_model_dir = "./output_model/student.pth"
        self.student_model_type = "resnet18"
        self.distill_epochs = 50
        self.adversaries = 120
        self.batch_size = 32
        self.learning_rate_generator = 0.2
        self.learning_rate_student = 2e-3
        self.latent_dim = 100
        self.img_size = 32
        self.channels = 1
        self.one_hot_loss_alpha = 1
        self.information_entropy_loss_beta = 5
        self.activation_loss_gamma = 0.1
        if config_path != "":
            self.readconfig(config_path)
    def readconfig(self,config_path):
        with open(config_path,encoding="utf-8") as config_file:
            config = yaml.load(config_file,Loader=yaml.FullLoader)
            self.classes = config["classes"]
            dafl_config = config["DAFL"]
            self.teacher_model_dir = dafl_config["teacher_model_dir"]
            self.teacher_model_structured = dafl_config["teacher_model_structured"]
            self.teacher_model_type = dafl_config["teacher_model_type"]
            self.student_model_dir = dafl_config["student_model_dir"]
            self.student_model_type = dafl_config["student_model_type"]
            self.distill_epochs = dafl_config["distill_epochs"]
            self.adversaries = dafl_config["adversaries"]
            self.batch_size = dafl_config["batch_size"]
            self.learning_rate_generator = dafl_config["learning_rate_generator"]
            self.learning_rate_student = float(dafl_config["learning_rate_student"])
            self.latent_dim = dafl_config["latent_dim"]
            self.img_size = dafl_config["img_size"]
            self.channels = dafl_config["channels"]
            self.one_hot_loss_alpha = dafl_config["one_hot_loss_alpha"]
            self.information_entropy_loss_beta = dafl_config["information_entropy_loss_beta"]
            self.activation_loss_gamma = dafl_config["activation_loss_gamma"]

    def distillate(self):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        data_test = CIFAR10("./data",
                             train=False,
                             transform=transform_test)
        data_test_loader = DataLoader(data_test, batch_size=self.batch_size, num_workers=0)
        best_acc_epoch = 0
        best_acc = 0

        generator = Generator(self.img_size, self.latent_dim, self.channels).to(device=self.device)
        if not self.teacher_model_structured:
            if self.teacher_model_type == "resnet18":
                teacher_model = torchvision.models.resnet18(num_classes=self.classes)
            if self.teacher_model_type == "resnet34":
                teacher_model = torchvision.models.resnet34(num_classes=self.classes)
            if self.teacher_model_type == "resnet50":
                teacher_model = torchvision.models.resnet50(num_classes=self.classes)
            if self.teacher_model_type == "resnet101":
                teacher_model = torchvision.models.resnet101(num_classes=self.classes)
            if self.teacher_model_type == "resnet152":
                teacher_model = torchvision.models.resnet152(num_classes=self.classes)
            if self.teacher_model_type == "mobilenet_v2":
                teacher_model = torchvision.models.mobilenet_v2(num_classes=self.classes)
            if self.teacher_model_type == "shufflenet_v2_x0_5":
                teacher_model = torchvision.models.shufflenet_v2_x0_5(num_classes=self.classes)
            if self.teacher_model_type == "alexnet":
                teacher_model = torchvision.models.alexnet(num_classes=self.classes)
            teacher_model = teacher_model.to(device=self.device)
        else:
            teacher_model = torch.load(self.teacher_model_dir)
            teacher_model = teacher_model.to(device=self.device)

        teacher_model.eval()

        hard_loss = torch.nn.CrossEntropyLoss().to(device=self.device)

        teacher = nn.DataParallel(teacher_model)
        teacher = teacher.module

        # generator = torch.load(self.student_model_dir[:-11]+"generator.pth")

        generator = nn.DataParallel(generator)

        student_model = torchvision.models.resnet18().to(device=self.device)


        if self.student_model_type == "resnet18":
            student_model = torchvision.models.resnet18(num_classes=self.classes).to(device=self.device)
        if self.student_model_type == "mobilenet_v2":
            student_model = torchvision.models.mobilenet_v2(num_classes=self.classes).to(device=self.device)
        if self.student_model_type == "shufflenet_v2_x0_5":
            student_model = torchvision.models.shufflenet_v2_x0_5(num_classes=self.classes).to(device=self.device)
        if self.student_model_type == "alexnet":
            student_model = torchvision.models.alexnet(num_classes=self.classes).to(device=self.device)

        # student_model = torch.load(self.student_model_dir)
        student = nn.DataParallel(student_model)

        # Optimizers
        optimizer_generator = torch.optim.SGD(generator.parameters(), lr=self.learning_rate_generator)
        optimizer_student = torch.optim.SGD(student.parameters(), lr=self.learning_rate_student, momentum=0.9, weight_decay=5e-4)

        best_model = student
        best_kd_loss = 2
        best_epoch = 0

        for epoch in range(self.distill_epochs):
            adjust_learning_rate(optimizer_student, epoch, self.learning_rate_student)
            adjust_learning_rate(optimizer_generator, epoch, self.learning_rate_generator)

            for i in range(self.adversaries):# 每个epoch对抗次数

                student.train()
                z = Variable(torch.randn(self.batch_size, self.latent_dim)).to(device=self.device)
                optimizer_generator.zero_grad()
                optimizer_student.zero_grad()
                gen_imgs = generator(z)
                outputs_teacher, features_teacher = forward_with_feature_output(teacher, gen_imgs)
                pred = outputs_teacher.data.max(1)[1]
                loss_activation = -features_teacher.abs().mean()
                loss_one_hot = hard_loss(outputs_teacher, pred)
                softmax_out_teacher = torch.nn.functional.softmax(outputs_teacher, dim=1).mean(dim=0)
                loss_information_entropy = (softmax_out_teacher * torch.log10(softmax_out_teacher)).sum()
                loss = loss_one_hot * self.one_hot_loss_alpha + loss_information_entropy * self.information_entropy_loss_beta + loss_activation * self.activation_loss_gamma
                loss_kd = kd_loss(student(gen_imgs.detach()), outputs_teacher.detach())
                loss += loss_kd
                loss.backward()
                optimizer_generator.step()
                optimizer_student.step()
                if i == self.adversaries-1:
                    print("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_kd: %f]" % (
                    epoch+1, self.distill_epochs, loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(),
                    loss_kd.item()))
                    if best_kd_loss > loss_kd.item():
                        best_model = student_model
                        best_epoch = epoch + 1
                        best_kd_loss = loss_kd.item()
                    print("[Best_kd_epoch %d] [best_kd_loss %f]" % (best_epoch, best_kd_loss))
                    torch.save(best_model, self.student_model_dir)
                    torch.save(generator, self.student_model_dir[:-11]+"generator.pth")

            test_acc = 0
            test_loss = 0.0
            test_data_size = 0
            with torch.no_grad():
                for i, (images, labels) in enumerate(data_test_loader):
                    images = images.cuda()
                    labels = labels.cuda()
                    student.eval()
                    output = student(images)
                    loss = hard_loss(output, labels)
                    outputs = nn.functional.softmax(output, dim=1)
                    pred = torch.argmax(outputs, dim=1)
                    acc = torch.sum(pred == labels)
                    test_loss += loss.item()
                    test_acc += acc.item()
                    test_data_size += labels.size(0)

            test_loss_epoch = test_loss / test_data_size
            test_acc_epoch = test_acc / test_data_size
            print('Test. Loss: %f, Accuracy: %f' % (test_loss_epoch, test_acc_epoch))
            if best_acc < test_acc_epoch:
                best_acc = test_acc_epoch
                best_acc_epoch = epoch + 1
            print('best_acc: %f, best_acc_epoch: %d' % (best_acc, best_acc_epoch))

        return best_model

if __name__ == "__main__":
    dafl = DAFL("config.yaml")
    best_model = dafl.distillate()
