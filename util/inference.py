# 预测类
import torch
from PIL.Image import Image
from torchvision.transforms import transforms

class Inference:
    def __init__(self,model_path):
        self.labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        #model_path = 'checkpoints/best_model.pth'
        # 加载模型
        self.model = torch.load(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    # 预测
    def predict(self, img_path):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img = Image.open(img_path)
        img = img.convert('L')
        img = transform(img)
        img = img.view(1, 1, 28, 28).to(self.device)
        output = self.model(img)
        output = torch.softmax(output, dim=1)
        # 每个预测值的概率
        probability = output.cpu().detach().numpy()[0]
        # 找出最大概率值的索引
        output = torch.argmax(output, dim=1)
        index = output.cpu().numpy()[0]
        # 预测结果
        pred = self.labels[index]
        #print(pred, probability[index])
        return pred,probability[index]