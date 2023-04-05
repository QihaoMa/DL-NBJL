from PIL import Image
from torch.utils.data import Dataset
import glob,os
import numpy as np


class WeldDataset(Dataset):



    def __init__(self, data_path, mode, image_transform, label_transform, pre):
        # 初始化函数，读取所有data_path下的图片
        self.mode = mode.lower()
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.pre = pre
        if self.mode == 'train':
            self.imgs_path = glob.glob(os.path.join(data_path, 'train/image/*.bmp'))
        else:
            self.imgs_path = glob.glob(os.path.join(data_path, 'val/image/*.bmp'))

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # 更新标签文件的逻辑
        label_path = label_path.replace('.bmp', '.bmp')

        # 读取训练图片和标签图片
        if self.pre:
            image = self.Gama(img_path=image_path,c=2,gama=0.8)
        else:
            image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)
        # 定义一个灰度变换
        # 转化
        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return image, label



    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

    def Gama(self, img_path, c=2, gama=0.8):
        '''
            伽马变换
            :param input_image: 原图像地址
            :param c: 伽马变换超参数
            :param gamma: 伽马值
            :return: 伽马变换后的图像
            '''
        image = Image.open(img_path)
        image = np.array(image).astype('uint8')
        img_norm = image / 255.0  # 注意255.0得采用浮点数
        img_gamma = c * np.power(img_norm, gama) * 255.0
        img_gamma[img_gamma > 255] = 255
        img_gamma[img_gamma < 0] = 0
        img_gamma = Image.fromarray(img_gamma, mode='RGB')
        return img_gamma