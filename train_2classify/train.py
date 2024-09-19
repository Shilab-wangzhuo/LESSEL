## 
from torchvision import datasets,transforms
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import os
import time
import argparse
import matplotlib.pyplot as plt


device="cuda" if torch.cuda.is_available() else "cpu"

class Efficientnet_train():
    def __init__(self,opt):
        self.losses = []
        self.valid_acc = []
        self.train_acc = []
        self.epochs=opt.epochs #训练周期
        self.batch_size=opt.batch_size #batch_size
        self.class_num=opt.class_num #类别数
        self.imgsz=opt.imgsz #图片尺寸
        self.img_dir=opt.img_dir #图片路径
        self.weights=opt.weights #模型路径
        self.save_dir=opt.save_dir #保存模型路径
        self.lr=opt.lr #初始化学习率
        self.moment=opt.m #动量
        self.start_time=(time.strftime("%m%d_%H%M",time.localtime()))
        
        # 模型加载
        base_model = EfficientNet.from_name('efficientnet-b4') #记载模型，使用b几的就改为b几
        state_dict = torch.load(self.weights)
        base_model.load_state_dict(state_dict)
        # 修改全连接层
        num_ftrs = base_model._fc.in_features
        base_model._fc = nn.Sequential(nn.Linear(num_ftrs, self.class_num),
                                       nn.Softmax(dim=1)  # 在全连接层后添加 softmax 激活函数
                                       )
        self.model = base_model.to(device)
        
        # 交叉熵损失函数
        self.cross = nn.CrossEntropyLoss()
        # 优化器
        self.optimzer = optim.SGD((self.model.parameters()), lr=self.lr, momentum=self.moment, weight_decay=0.0004)

        #获取处理后的数据集和类别映射表
        self.trainx,self.valx,self.b=self.process()
        print(self.b)
    def __call__(self):
        best_acc = 0
        self.model.train(True)
        self.start_time=(time.strftime("%m%d_%H%M",time.localtime()))
        for ech in range(self.epochs):
            optimzer1 = self.lrfn(ech, self.optimzer)

            print("----------Start Train Epoch %d----------" % (ech + 1))
            # 开始训练
            run_loss = 0.0  # 损失
            run_correct = 0.0  # 准确率
            count = 0.0  # 分类正确的个数

            for i, data in enumerate(self.trainx):

                inputs, label = data
                inputs, label = inputs.to(device), label.to(device)

                # 训练
                optimzer1.zero_grad()
                output = self.model(inputs)

                loss = self.cross(output, label)
                loss.backward()
                optimzer1.step()

                run_loss += loss.item()  # 损失累加
                
                _, pred = torch.max(output.data, 1)
                count += label.size(0)  # 求总共的训练个数
                run_correct += pred.eq(label.data).cpu().sum()  # 截止当前预测正确的个数
                #每隔100个batch打印一次信息，这里打印的ACC是当前预测正确的个数/当前训练过的的个数
                if (i+1)%100==0:
                    print('[Epoch:{}__iter:{}/{}] | Acc:{}'.format(ech + 1,i+1,len(self.trainx), run_correct/count))

            train_acc = run_correct / count
            # 每次训完一批打印一次信息
            print('Epoch:{} | Loss:{} | Acc:{}'.format(ech + 1, run_loss / len(self.trainx), train_acc))
            epoch_loss = run_loss / len(self.trainx)
            self.losses.append(epoch_loss)
            self.train_acc.append(train_acc)
                
            
            # 训完一批次后进行验证
            print("----------Waiting Test Epoch {}----------".format(ech + 1))
            with torch.no_grad():
                correct = 0.  # 预测正确的个数
                total = 0.  # 总个数
                for inputs, labels in self.valx:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.model(inputs)

                    # 获取最高分的那个类的索引
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += pred.eq(labels).cpu().sum()
                test_acc = correct / total
                self.valid_acc.append(test_acc)
                print("批次%d的验证集准确率" % (ech + 1), correct / total)
            
            save_weight=self.save_dir+os.sep+self.start_time #保存路径
            os.makedirs(save_weight,exist_ok=True)
            weight_name = "Epoch" + str(ech)+".pth"
            torch.save(self.model, save_weight + os.sep + weight_name)
            
            '''if best_acc < test_acc:
                best_acc = test_acc
                #start_time=(time.strftime("%m%d_%H%M",time.localtime()))
                save_weight=self.save_dir+os.sep+start_time #保存路径
                os.makedirs(save_weight,exist_ok=True)
                torch.save(self.model, save_weight + os.sep + "best.pth")'''
        self.plot_loss_curve()
        self.plot_accuracy_curve()
        
    def plot_loss_curve(self):
        plt.plot(range(1, len(self.losses) + 1), self.losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        save_weight=self.save_dir+os.sep+self.start_time+os.sep+'Loss_Curve'
        plt.savefig(save_weight)
        plt.show()

    def plot_accuracy_curve(self):
        plt.plot(range(1, len(self.train_acc) + 1), self.train_acc, 'o-', label='Training Accuracy')  # 散点图形式
        plt.plot(range(1, len(self.valid_acc) + 1), self.valid_acc, label='Validation Accuracy')  # 直线形式
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy Curve')
        plt.legend()
        save_weight=self.save_dir+os.sep+self.start_time+os.sep+'Accuracy_Curve'
        plt.savefig(save_weight)
        plt.show()

    
    #数据处理
    def process(self):
        # 数据增强
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((self.imgsz, self.imgsz)),  # resize
                transforms.CenterCrop((self.imgsz, self.imgsz)),  # 中心裁剪
                transforms.RandomRotation(30),  # 随机旋转，旋转范围为【-10,10】
                transforms.RandomHorizontalFlip(p=0.2),  # 水平镜像
                #transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.2, hue=0.2),
                transforms.ToTensor(),  # 转换为张量
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
            ]),
            "val": transforms.Compose([
                transforms.Resize((self.imgsz, self.imgsz)),  # resize
                transforms.CenterCrop((self.imgsz, self.imgsz)),  # 中心裁剪
                transforms.ToTensor(),  # 张量转换
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

        # 定义图像生成器
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.img_dir, x), data_transforms[x]) for x in
                          ['train', 'val']}
        # 得到训练集和验证集
        trainx = DataLoader(image_datasets["train"], batch_size=self.batch_size, shuffle=True, drop_last=True)
        valx = DataLoader(image_datasets["val"], batch_size=self.batch_size, shuffle=True, drop_last=True)

        b = image_datasets["train"].class_to_idx  # id和类别对应

        return trainx,valx,b


    # 学习率慢热加下降
    def lrfn(self,num_epoch, optimzer):
        lr_start = 0.00001  # 初始值
        max_lr = 0.0001  # 最大值
        lr_up_epoch = 10  # 学习率上升10个epoch
        lr_sustain_epoch = 5  # 学习率保持不变
        lr_exp = .9 # 衰减因子
        if num_epoch < lr_up_epoch:  # 0-10个epoch学习率线性增加
            lr = (max_lr - lr_start) / lr_up_epoch * num_epoch + lr_start
        elif num_epoch < lr_up_epoch + lr_sustain_epoch:  # 学习率保持不变
            lr = max_lr
        else:  # 指数下降
            lr = (max_lr - lr_start) * lr_exp ** (num_epoch - lr_up_epoch - lr_sustain_epoch) + lr_start
        for param_group in optimzer.param_groups:
            param_group['lr'] = lr
        return optimzer

#参数设置
def make_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--weights",type=str,default="train_2classify/pretrain_model/efficientnet-b4-6ed6700e.pth",help='initial weights path')#预训练模型路径
    parser.add_argument("--img-dir",type=str,default="train_2classify/datasetB",help="dataset path") #数据集的路径
    parser.add_argument("--imgsz",type=int,default=224,help="image size") #图像尺寸
    parser.add_argument("--epochs",type=int,default=10,help="train epochs")#训练批次
    parser.add_argument("--batch-size",type=int,default=16,help="train batch-size") #batch-size
    parser.add_argument("--class_num",type=int,default=2,help="class num") #类别数
    parser.add_argument("--lr",type=float,default=0.00001,help="Init lr") #学习率初始值
    parser.add_argument("--m",type=float,default=0.9,help="optimer momentum") #动量
    parser.add_argument("--save-dir",type=str,default="train_2classify/weightB",help="save models dir")#保存模型路径
    #opt=parser.parse_known_args()[0]
    return parser
    
if __name__ == '__main__':
    opt = make_parser().parse_args()
    models=Efficientnet_train(opt)
    models()
    