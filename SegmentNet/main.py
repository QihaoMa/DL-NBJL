import os

from thop import profile
from ptflops import get_model_complexity_info
import time
import torch.utils.data as data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm

import transforms as ext_transforms
from data.WeldDataset import WeldDataset
from utils import *
from models.ENet.enet import ENet
from models.UNet.unet import UNet
from models.FastSCNN.fast_scnn import FastSCNN
from models.LiteSegMobileNet.liteseg_mobilenet import LiteSegMobileNet
from args import get_arguments
from metric.iou import IoU
from train import Train
from test import Test
from torchvision.transforms import InterpolationMode


args = get_arguments()

def GetTransforms(isTrain=True):
    if isTrain:
        image_transform = transforms.Compose([
            # 亮度 对比度
            transforms.ColorJitter(brightness=[0.4,1.4],contrast=[0.5,1.5]),
            transforms.Resize((args.crop_size[0], args.crop_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.522, 0.522, 0.522], std=[0.125, 0.125, 0.125])
        ])

        label_transform = transforms.Compose([
            transforms.Resize((args.crop_size[0], args.crop_size[1]), interpolation=InterpolationMode.NEAREST),
            ext_transforms.PILToLongTensor()
        ])
    else:
        image_transform = transforms.Compose([
            transforms.Resize((args.crop_size[0], args.crop_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.503, 0.522, 0.522], std=[0.138, 0.138, 0.138])
        ])

        label_transform = transforms.Compose([
            transforms.Resize((args.crop_size[0], args.crop_size[1]), interpolation=InterpolationMode.NEAREST),
            ext_transforms.PILToLongTensor()
        ])
    return image_transform,label_transform


def loadData(run_dir):

    train_image_transform,train_label_transform = GetTransforms(isTrain=True)

    # 数据集
    train_dataset = WeldDataset(data_path=args.dataset,
                mode='train',
                image_transform=train_image_transform,
                label_transform=train_label_transform,
                pre=args.preprocess)
    val_image_transform, val_label_transform = GetTransforms(isTrain=True)
    val_dataset = WeldDataset(data_path=args.dataset,
                              mode='val',
                              image_transform=val_image_transform,
                              label_transform=val_label_transform,
                              pre=args.preprocess)
    # mean,std = compute_mean_std(train_dataset)
    # print('T均值：{}'.format(mean))
    # print('T方差：{}'.format(std))
    # mean, std = compute_mean_std(val_dataset)
    # print('V均值：{}'.format(mean))
    # print('V方差：{}'.format(std))
    # loader
    train_loader = data.DataLoader(train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=1)
    val_loader = data.DataLoader(val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=1)


    # 加载类别权重
    class_weights = enet_weighing(train_loader, args.num_classes)
    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float()
    
    # 打印输出数据集情况
    print("类别权重: ", class_weights)
    print("训练数据集数量:{}".format(len(train_dataset)))
    print("验证数据集数量:{}".format(len(val_dataset)))
    with open(os.path.join(run_dir,'train_param.txt'), 'a') as f:
        f.writelines('类别权重' + ' : ' + str(class_weights) + '\n')
        f.writelines('训练集：' + ' : ' + str(len(train_dataset)) + '\n')
        f.writelines('验证集：' + ' : ' + str(len(val_dataset)) + '\n')
        f.close()

    return train_loader,val_loader,class_weights


def train(run_dir,train_loader, val_loader, class_weights, device):
    model = None
    # 模型初始化
    if args.model.lower() == 'fastscnn':
        model = FastSCNN(num_classes=args.num_classes).to(device)
    elif args.model.lower() == 'enet':
        model = ENet(num_classes=args.num_classes,channels=[8,32,64],factor=1,sp_layer_mid=False,sp_layer_bottle=True).to(device)
    elif args.model.lower() == 'unet':
        model = UNet(in_channels=3,num_classes=args.num_classes).to(device)
    elif args.model.lower() == 'litesegmobilenet':
        model = LiteSegMobileNet(num_classes=args.num_classes).to(device)
    # 迁移学习，加载预训练模型
    if args.pre_train:
        PTH = torch.load(args.pre_train_path,map_location=device)
        model.load_state_dict(PTH)
    # 定义损失函数
    if args.loss.lower() == 'celoss':
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    elif args.loss.lower() == 'focalloss':
        criterion = FocalLoss(weight=torch.FloatTensor(class_weights).to(device),gamma=args.gama)

    # ************************************
    

    # 损失优化函数
    if  args.optimizer.lower() == 'adam':
        # Adam
        optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay)
        # 学习率衰减方法
        # lr_updater = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
        lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, args.lr_decay)
    elif args.optimizer.lower() == 'sgd':
        # SGD
        # 参数列表
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, args.lr_decay)
        # lr_updater = lr_scheduler.MultiStepLR(
        #     optimizer=optimizer,
        #     milestones=[10,30,50,75,105],
        #     gamma=0.7
        # )
        # lr_updater = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    # ************************************



    
    # *************************************
    # 定义评估指标
    metric = IoU(args.num_classes)
    # 执行训练
    trainer = Train(model, train_loader, optimizer, criterion, metric, device)
    tester = Test(model, val_loader, criterion, metric, device)
    min_loss = float('inf')
    min_val_loss = float('inf')
    best_miou = 0
    best_iou = None
    logwriter = SummaryWriter(run_dir+'/log')
    tags = ["train_loss",
            "val_loss",
            "train_miou",
            "val_miou",
            "learning_rate",
            "train_acc",
            "val_acc"]
    model_save_path = os.path.join(run_dir,'model')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    start_epoch = -1
    if args.resume:
        CKPT_PATH = r'run/ENet/SGD_MultiLR/exp_2022-12-04-21-33-19/model/ckpt_best.pth'
        ckpt = torch.load(CKPT_PATH,map_location=device)
        model.load_state_dict(ckpt['model'],strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        lr_updater.load_state_dict(ckpt['lr_updater'])
    # 保存参数
    flops, params = get_model_complexity_info(model,(3,args.crop_size[0], args.crop_size[1]),as_strings=True,print_per_layer_stat=False)
    with open(os.path.join(run_dir,'train_result.txt'), 'a') as f:
        f.writelines('model input size:[{},{}]'.format(args.crop_size[0], args.crop_size[1]) + '\n')
        f.writelines('计算量: {}'.format(flops) + '\n')
        f.writelines('参数量: {}'.format(params) + '\n')
        f.close()
    # 开始训练
    with tqdm(total=args.epochs) as pbar:
        for epoch in range(start_epoch+1, args.epochs):
            pbar.set_description(desc="Epoch: {0:d}/{1:d}".format(epoch, args.epochs))

            train_epoch_loss, (train_iou, train_miou, train_acc, train_macc) = trainer.run_epoch(print_step_loss=True)
            lr_updater.step()

            print(">>>> {3:5}: Epoch: {0:d} >>>> Avg.loss: {1:.4f} >>>> Mean IoU: {2:.4f} >>>> Class-Iou: {4:40} >>>> Accuracy: {5:.4f} >>>> Class-Accuracy: {6:40}".
                  format(epoch, train_epoch_loss, train_miou, 'Train', str(train_iou), train_macc, str(train_acc)))
            val_epoch_loss, (val_iou, val_miou, val_acc, val_macc) = tester.run_epoch(print_step_loss=False)

            print(">>>> {3:5}: Epoch: {0:d} >>>> Avg.loss: {1:.4f} >>>> Mean IoU: {2:.4f} >>>> Class-Iou: {4:40} >>>> Accuracy: {5:.4f} >>>> Class-Accuracy: {6:40}".
                  format(epoch, val_epoch_loss, val_miou, 'val', str(val_iou), val_macc, str(val_acc)))
            # 添加到日志中
            logwriter.add_scalar(tags[0], train_epoch_loss, epoch)
            logwriter.add_scalar(tags[1], val_epoch_loss, epoch)
            logwriter.add_scalar(tags[2], train_miou, epoch)
            logwriter.add_scalar(tags[3], val_miou, epoch)
            logwriter.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
            logwriter.add_scalar(tags[5], train_macc, epoch)
            logwriter.add_scalar(tags[6], val_macc, epoch)

            if train_epoch_loss < min_loss:
                min_loss = train_epoch_loss
                torch.save(model.state_dict(), model_save_path+'/best_trainloss_model.pth')
            if val_miou > best_miou:
                best_miou = val_miou
                best_iou = val_iou
                torch.save(model.state_dict(), model_save_path+'/best_valIou_model.pth')
            if val_epoch_loss < min_val_loss:
                min_val_loss = val_epoch_loss
                torch.save(model.state_dict(), model_save_path+'/best_valloss_model.pth')
            if (epoch % 10 == 0):
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'lr_updater': lr_updater.state_dict()
                }
                torch.save(checkpoint, model_save_path + '/ckpt_best.pth')
            pbar.update(1)
    logwriter.close()
    with open(os.path.join(run_dir,'train_result.txt'), 'a') as f:
        f.writelines('best val MIou:%.2f%%' % (best_miou) + '\n')
        f.writelines('best val iou:{}'.format(str(val_iou)) + '\n')
        f.writelines('train min loss:{}'.format(min_loss) + '\n')
        f.writelines('val min loss:{}'.format(min_val_loss) + '\n')
        f.close()

def saveParams(path):
    argsDict = args.__dict__
    with open(os.path.join(path,'train_param.txt'), 'a') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.close()



if __name__ == '__main__':
    ROOT_Path = os.getcwd()
    print(ROOT_Path)
    run_dir = os.path.join(ROOT_Path,'run/SPB/ENet_Half_BottleSP')
    startTime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print('Start Time:{}'.format(str(startTime)))
    run_dir = os.path.join(run_dir,'exp_{}'.format(str(startTime)))
    print(run_dir)
    os.makedirs(run_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, class_weights = loadData(run_dir)
    saveParams(run_dir)
    train(run_dir,train_loader, val_loader, class_weights, device)
    endTime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print('End Time:{}'.format(str(endTime)))
