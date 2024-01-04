# 导入所需的库
import torch as tc
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os, sys, cv2
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models_pytorch as smp

# 导入 Albumentations 的 PyTorch 转换模块
from albumentations.pytorch import ToTensorV2

# 导入 PyTorch 数据集和数据加载工具
from torch.utils.data import Dataset, DataLoader

# 导入 PyTorch 的 DataParallel 模块
from torch.nn.parallel import DataParallel # 单机多卡的分布式训练（数据并行） 模型训练加速

# 导入文件和路径处理工具
from glob import glob

import torch.nn.functional as F

########################################################################################
# 显存： 骨干网络的复杂度 vs 输入尺寸 vs 批大小
# 【理想情况】：模型输入1024 * 1024，1500 * 1500

# 前20 epoch 512 * 512  训练模型
# 后20 epoch 1024 * 1024 继续训练

class CFG:
    # ============== 预测目标 =============
    target_size = 1

    # ============== 模型配置 =============
    model_name = 'Unet'
    backbone = 'se_resnext50_32x4d'

    in_chans = 5  # 输入通道数, 我感觉是5张图片看做一个样本

    # ============== 训练配置 =============
    image_size = 1024 # 512/1024  # 图片大小
    input_size = 1024 # 512/1024   # 输入尺寸

    train_batch_size = 16  # 训练批量大小
    valid_batch_size = train_batch_size * 2  # 验证批量大小

    epochs = 20 # 20/40  # 训练轮数
    
    lr = 6e-5  # 学习率
    chopping_percentile = 1e-3  # 切割百分比
    # ============== 折数 =============
    valid_id = 1  # 验证集编号

    # ============== 数据增强 =============
    train_aug_list = [
        A.Rotate(limit=45, p=0.5),  # 旋转
        A.RandomScale(scale_limit=(0.8, 1.25), interpolation=cv2.INTER_CUBIC, p=0.5),  # 随机缩放
        A.RandomCrop(input_size, input_size, p=1),  # 随机裁剪
        A.RandomGamma(p=0.75),  # 随机Gamma变换
        A.RandomBrightnessContrast(p=0.5, ),  # 随机亮度对比度变换
        A.GaussianBlur(p=0.5),  # 高斯模糊
        A.MotionBlur(p=0.5),  # 运动模糊
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),  # 网格扭曲
        ToTensorV2(transpose_mask=True),  # 转换为张量
    ]
    train_aug = A.Compose(train_aug_list)
    valid_aug_list = [
        ToTensorV2(transpose_mask=True),  # 转换为张量
    ]
    valid_aug = A.Compose(valid_aug_list)

########################################################################################
class CustomModel(nn.Module):
    def __init__(self, CFG, weight=None):
        super().__init__()
        
        # 初始化模型，使用了 segmentation_models_pytorch 库中的 Unet 模型
        self.model = smp.Unet(
            encoder_name=CFG.backbone, 
            encoder_weights=weight,
            in_channels=CFG.in_chans,
            classes=CFG.target_size,
            activation=None,
        )

    def forward(self, image):
        # 模型的前向传播
        output = self.model(image)
        # 如果需要，可以在这里对输出进行额外的处理
        # output = output.squeeze(-1)
        return output[:, 0]  # 选择输出的第一个通道，这里假设输出是多通道的sigmoid()

def build_model(weight="imagenet"):
    # # 加载环境变量
    # load_dotenv()

    local_weights_path = '/home/xyli/kaggle/kaggle_vasculature/workplace/se_resnext50_32x4d-a260b3a4.pth'
    
    # 输出模型名称和使用的骨干网络
    print('model_name', CFG.model_name)
    print('backbone', CFG.backbone)

    # 构建并返回模型
    # model = CustomModel(CFG, weight)
    model = CustomModel(CFG, None)
    
    model.encoder_weights.load_state_dict(tc.load(local_weights_path)) # 加载本地权重文件

    return model.cuda()


########################################################################################

def min_max_normalization(x: tc.Tensor) -> tc.Tensor:
    """最小-最大归一化函数

    参数:
    x (tc.Tensor): 输入张量，形状为(batch, f1, ...)

    返回:
    tc.Tensor: 归一化后的张量，保持原始形状
    """
    # 获取输入张量的形状
    shape = x.shape

    # 如果输入张量的维度大于2，将其展平成二维张量
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    # 计算每行的最小值和最大值
    min_ = x.min(dim=-1, keepdim=True)[0]
    max_ = x.max(dim=-1, keepdim=True)[0]

    # 如果最小值的平均值为0，最大值的平均值为1，说明已经是归一化状态，直接返回
    if min_.mean() == 0 and max_.mean() == 1:
        return x.reshape(shape)

    # 进行最小-最大归一化处理
    x = (x - min_) / (max_ - min_ + 1e-9)
    return x.reshape(shape)

def norm_with_clip(x: tc.Tensor, smooth=1e-5):
    """带截断的标准化函数

    参数:
    x (tc.Tensor): 输入张量
    smooth (float): 平滑值，用于避免除零错误，默认为1e-5

    返回:
    tc.Tensor: 标准化后的张量
    """
    # 获取除第一维外的所有维度
    dim = list(range(1, x.ndim))
    
    # 计算均值和标准差
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)

    # 标准化处理
    x = (x - mean) / (std + smooth)

    # 对大于5和小于-3的值进行截断处理
    x[x > 5] = (x[x > 5] - 5) * 1e-3 + 5
    x[x < -3] = (x[x < -3] + 3) * 1e-3 - 3

    return x

def add_noise(x: tc.Tensor, max_randn_rate=0.1, randn_rate=None, x_already_normed=False):
    """
    给定输入张量 x，添加噪声并返回处理后的张量

    参数:
        - x: 输入张量，形状为 (batch, f1, f2, ...)
        - max_randn_rate: 随机噪声的最大比例，默认为 0.1
        - randn_rate: 可选参数，手动指定噪声比例，如果为 None 则随机生成
        - x_already_normed: 布尔值，指示输入张量是否已经进行了标准化

    返回:
        处理后的张量，其方差已被归一化

    注意:
        - 如果输入张量已经标准化 (x_already_normed=True)，则 x_std 为全 1 张量，x_mean 为全 0 张量。
        - 如果输入张量未标准化，根据输入的维度进行标准化处理。

    参考:
        - https://blog.csdn.net/chaosir1991/article/details/106960408
    """
    ndim = x.ndim - 1

    if x_already_normed:
        x_std = tc.ones([x.shape[0]] + [1] * ndim, device=x.device, dtype=x.dtype)
        x_mean = tc.zeros([x.shape[0]] + [1] * ndim, device=x.device, dtype=x.dtype)
    else:
        dim = list(range(1, x.ndim))
        x_std = x.std(dim=dim, keepdim=True)
        x_mean = x.mean(dim=dim, keepdim=True)

    if randn_rate is None:
        randn_rate = max_randn_rate * np.random.rand() * tc.rand(x_mean.shape, device=x.device, dtype=x.dtype)

    # 计算噪声缩放系数
    cache = (x_std ** 2 + (x_std * randn_rate) ** 2) ** 0.5 + 1e-7

    # 添加噪声并返回处理后的张量
    return (x - x_mean + tc.randn(size=x.shape, device=x.device, dtype=x.dtype) * randn_rate * x_std) / cache

class Data_loader(Dataset):
    def __init__(self, paths, is_label):
        # 初始化函数，接收数据集路径和是否为标签的参数
        self.paths = paths  # 数据集路径列表
        self.paths.sort()   # 对路径进行排序
        self.is_label = is_label  # 是否为标签数据
    
    def __len__(self):
        # 返回数据集的长度
        return len(self.paths)
    
    def __getitem__(self, index):
        # 获取数据集中指定索引的样本
        img = cv2.imread(self.paths[index], cv2.IMREAD_GRAYSCALE)  # 读取灰度图像
        img = tc.from_numpy(img)  # 将图像转换为PyTorch张量

        if self.is_label:
            # 如果是标签数据，将非零像素值设为255（二值化）
            img = (img != 0).to(tc.uint8) * 255
        else:
            # 如果不是标签数据，将图像转换为8位无符号整数类型
            img = img.to(tc.uint8)

        return img  # 返回处理后的图像

def load_data(paths, is_label=False):
    # 创建Data_loader对象，处理数据路径和是否为标签的标志
    data_loader = Data_loader(paths, is_label)
    # 创建DataLoader，设置批量大小为16，使用2个工作进程加载数据
    data_loader = DataLoader(data_loader, batch_size=16, num_workers=2)
    # 存储数据的列表
    data = []
    # 遍历数据加载器，将每个批次的数据添加到列表中
    # for x in tqdm(data_loader):
    #     data.append(x)
    for x in data_loader:
        data.append(x)
    
    # 将数据列表拼接为一个张量
    x = tc.cat(data, dim=0)
    # 释放内存，删除数据列表
    del data
    
    # 如果不是标签数据
    if not is_label:
        # 对数据进行百分比截断处理
        ########################################################################
        # 计算数据张量的阈值
        TH = x.reshape(-1).numpy()
        # 根据设定的百分比确定阈值位置
        index = -int(len(TH) * CFG.chopping_percentile)
        # 对阈值进行分区操作，并取得分区后的阈值
        TH: int = np.partition(TH, index)[index]
        # 将大于阈值的元素设置为阈值
        x[x > TH] = int(TH)
        ########################################################################
        # 重新计算数据张量的阈值
        TH = x.reshape(-1).numpy()
        # 根据设定的百分比确定阈值位置
        index = -int(len(TH) * CFG.chopping_percentile)
        # 对阈值进行分区操作，并取得分区后的阈值
        TH: int = np.partition(TH, -index)[-index]
        # 将小于阈值的元素设置为阈值
        x[x < TH] = int(TH)
        ########################################################################
        # 对数据进行最小-最大归一化，并将数据类型转换为uint8
        x = (min_max_normalization(x.to(tc.float16)[None])[0] * 255).to(tc.uint8)
    
    # 返回处理后的数据张量
    return x

#https://www.kaggle.com/code/kashiwaba/sennet-hoa-train-unet-simple-baseline
def dice_coef(y_pred: tc.Tensor, y_true: tc.Tensor, thr=0.5, dim=(-1, -2), epsilon=0.001):
    # 对预测值进行sigmoid激活，将其转换到0到1的范围
    y_pred = y_pred.sigmoid()
    
    # 将真实值转换为float32类型
    y_true = y_true.to(tc.float32)
    
    # 将预测值二值化，使用阈值thr，默认为0.5
    y_pred = (y_pred > thr).to(tc.float32)
    
    # 计算交集（intersection），即预测值和真实值同时为1的位置之和
    inter = (y_true * y_pred).sum(dim=dim)
    
    # 计算分母，即真实值和预测值中1的位置之和
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    
    # 计算Dice系数，加入平滑项epsilon以防分母为0
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean()
    
    # 返回Dice系数作为评估指标
    return dice

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        # 如果你的模型包含 sigmoid 或等效的激活层，请注释掉下面这行
        inputs = inputs.sigmoid()   
        
        # 将标签和预测张量展平
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算交集
        intersection = (inputs * targets).sum()                            
        
        # 计算 Dice 系数
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        # 返回 Dice 损失
        return 1 - dice

class Kaggld_Dataset(Dataset):
    def __init__(self, x: list, y: list, arg: bool = False):
        super(Dataset, self).__init__()
        self.x = x  # 输入图像列表，每个元素为形状为(C, H, W)的图像
        self.y = y  # 目标图像列表，每个元素为形状为(C, H, W)的图像
        self.image_size = CFG.image_size  # 图像大小
        self.in_chans = CFG.in_chans  # 输入通道数
        self.arg = arg  # 是否进行数据增强
        if arg:
            self.transform = CFG.train_aug  # 训练数据增强配置
        else:
            self.transform = CFG.valid_aug  # 验证数据增强配置

    def __len__(self) -> int:
        return sum([y.shape[0] - self.in_chans for y in self.y])

    def __getitem__(self, index):

        # 我感觉是处理不同kidney数据集之间的一些不足量的末尾数据
        i = 0
        for x in self.x:
            if index > x.shape[0] - self.in_chans:
                index -= x.shape[0] - self.in_chans
                i += 1
            else:
                break
        x = self.x[i] # 被赋值后的x是某一kidney所有图片集合
        y = self.y[i]


        # my code
        if x.shape[1] < self.image_size:
            gap = self.image_size - x.shape[1]
            # 按照 (left, right, top, bottom) 的顺序表示在四个方向上的填充大小
            padding_size = (0, 0, 0, gap+1) # 底部填充0
            x = F.pad(x, padding_size)
            y = F.pad(y, padding_size)

        if x.shape[2] < self.image_size:
            gap = self.image_size - x.shape[2]
            # 按照 (left, right, top, bottom) 的顺序表示在四个方向上的填充大小
            padding_size = (0, gap+1, 0, 0) # 填充右边0
            x = F.pad(x, padding_size)
            y = F.pad(y, padding_size)
            
        # # my code
        # if x.shape[1] < self.image_size or x.shape[2] < self.image_size:
        #     x = x[index:index + self.in_chans, :, :]
        #     y = y[index + self.in_chans // 2, :, :]
        # else:
        #     x_index = np.random.randint(0, x.shape[1] - self.image_size)
        #     y_index = np.random.randint(0, x.shape[2] - self.image_size)

        #     x = x[index:index + self.in_chans, x_index:x_index + self.image_size, y_index:y_index + self.image_size]
        #     y = y[index + self.in_chans // 2, x_index:x_index + self.image_size, y_index:y_index + self.image_size]


        x_index = np.random.randint(0, x.shape[1] - self.image_size)
        y_index = np.random.randint(0, x.shape[2] - self.image_size)

        x = x[index:index + self.in_chans, x_index:x_index + self.image_size, y_index:y_index + self.image_size]
        y = y[index + self.in_chans // 2, x_index:x_index + self.image_size, y_index:y_index + self.image_size]

        # 进行数据增强
        data = self.transform(image=x.numpy().transpose(1, 2, 0), mask=y.numpy())
        x = data['image']
        y = data['mask'] >= 127

        if self.arg:
            i = np.random.randint(4)
            x = x.rot90(i, dims=(1, 2))
            y = y.rot90(i, dims=(0, 1))
            for i in range(3):
                if np.random.randint(2):
                    x = x.flip(dims=(i,))
                    if i >= 1:
                        y = y.flip(dims=(i - 1,))
        return x, y  # 返回处理后的图像数据，类型为(uint8, uint8)

########################################################################################

if __name__=='__main__':

    train_x = [] # train_x=[[all pic of kidney_1_dense], [all pic of kidney_1_voi]...]
    train_y = []

    # 数据集根路径
    # root_path = "/root/autodl-tmp/blood-vessel-segmentation/"
    root_path = "/home/xyli/kaggle/blood-vessel-segmentation/"
    
    # 数据集中子路径
    paths = [
                # "/root/autodl-tmp/blood-vessel-segmentation/train/kidney_1_dense",
                # # "/root/autodl-tmp/blood-vessel-segmentation/train/kidney_1_voi",
                # # "/root/autodl-tmp/blood-vessel-segmentation/train/kidney_2",
                # # "/root/autodl-tmp/blood-vessel-segmentation/train/kidney_3_dense",
                # # "/root/autodl-tmp/blood-vessel-segmentation/train/kidney_3_sparse"

                "/home/xyli/kaggle/blood-vessel-segmentation/train/kidney_1_dense",
                # "/home/xyli/kaggle/blood-vessel-segmentation/train/kidney_1_voi",
                # "/home/xyli/kaggle/blood-vessel-segmentation/train/kidney_2",
                # "/home/xyli/kaggle/blood-vessel-segmentation/train/kidney_3_dense",
                # "/home/xyli/kaggle/blood-vessel-segmentation/train/kidney_3_sparse"

            ]

    # 遍历子路径
    for i, path in enumerate(paths):
        # 排除特定路径
        # if path == "/root/autodl-tmp/blood-vessel-segmentation/train/kidney_3_dense":
        if path == "/home/xyli/kaggle/blood-vessel-segmentation/train/kidney_3_dense":    
            continue
        
        # 加载图像数据（非标签）
        x = load_data(glob(f"{path}/images/*"), is_label=False)
        print("train dataset x shape:", x.shape)
        
        # 加载标签数据
        y = load_data(glob(f"{path}/labels/*"), is_label=True)
        print("train dataset y shape:", y.shape)
        
        # 将数据添加到训练集
        train_x.append(x)
        train_y.append(y)

        # 数据维度变换及数据增强
        #(C,H,W)
        #augmentation
        train_x.append(x.permute(1, 2, 0))
        train_y.append(y.permute(1, 2, 0))
        train_x.append(x.permute(2, 0, 1))
        train_y.append(y.permute(2, 0, 1))

    # 验证集路径
    # path1 = "/root/autodl-tmp/blood-vessel-segmentation/train/kidney_3_sparse"
    # path2 = "/root/autodl-tmp/blood-vessel-segmentation/train/kidney_3_dense"

    path1 = "/home/xyli/kaggle/blood-vessel-segmentation/train/kidney_3_sparse"
    path2 = "/home/xyli/kaggle/blood-vessel-segmentation/train/kidney_3_dense"

    # 获取验证集图像和标签路径列表
    paths_y = glob(f"{path2}/labels/*")
    paths_x = [x.replace("labels", "images").replace("dense", "sparse") for x in paths_y]

    # 加载验证集图像和标签数据
    val_x = load_data(paths_x, is_label=False)
    print("validate dataset x shape:", val_x.shape)
    val_y = load_data(paths_y, is_label=True)
    print("validate dataset y shape:", val_y.shape)	




    ########################################################################################

    # 启用cudnn加速，并进行基准测试
    tc.backends.cudnn.enabled = True
    tc.backends.cudnn.benchmark = True
        
    # 创建训练数据集对象，使用Kaggld_Dataset类，传入训练数据和标签，arg=True表示进行一些额外的操作
    train_dataset = Kaggld_Dataset(train_x, train_y, arg=True)
    # 创建训练数据加载器，设置批大小、工作线程数、是否打乱数据、是否将数据存储在固定内存中
    train_dataset = DataLoader(train_dataset, batch_size=CFG.train_batch_size, num_workers=2, shuffle=True, pin_memory=True)

    # 创建验证数据集对象，使用Kaggld_Dataset类，传入验证数据和标签
    val_dataset = Kaggld_Dataset([val_x], [val_y])
    # 创建验证数据加载器，设置批大小、工作线程数、是否打乱数据、是否将数据存储在固定内存中
    val_dataset = DataLoader(val_dataset, batch_size=CFG.valid_batch_size, num_workers=2, shuffle=False, pin_memory=True)

    # 构建模型
    model = build_model()
    # 使用DataParallel进行模型并行处理
    model = DataParallel(model)

    # 使用DiceLoss作为损失函数
    loss_fc = DiceLoss()
    # 使用AdamW优化器，传入模型参数和学习率
    optimizer = tc.optim.AdamW(model.parameters(), lr=CFG.lr)

    # 使用GradScaler进行梯度缩放，用于混合精度训练 2080 3090 / 1080ti
    scaler = tc.cuda.amp.GradScaler()

    # 设置学习率调度器，使用OneCycleLR策略
    scheduler = tc.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.lr,
                                                steps_per_epoch=len(train_dataset), epochs=CFG.epochs+1,
                                                pct_start=0.1)

    print("start the train!")
    # 循环训练模型
    for epoch in range(CFG.epochs):
        model.train()
        
    #     # 创建进度条以显示训练进度
    #     time = tqdm(range(len(train_dataset)))
        
        losss = 0
        scores = 0
        
        # 遍历训练数据集
        for i, (x, y) in enumerate(train_dataset):
            x = x.cuda().to(tc.float32)
            y = y.cuda().to(tc.float32)
            
            # 数据预处理
            x = norm_with_clip(x.reshape(-1, *x.shape[2:])).reshape(x.shape)
            x = add_noise(x, max_randn_rate=0.5, x_already_normed=True)
            
            # 使用自动混合精度进行前向传播和损失计算
            with autocast():
                pred = model(x)
                loss = loss_fc(pred, y)
            
            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            # 计算并更新平均损失和分数
            score = dice_coef(pred.detach(), y)
            losss = (losss * i + loss.item()) / (i + 1)
            scores = (scores * i + score) / (i + 1)
            
    #         # 更新进度条
    #         time.set_description(f"epoch:{epoch},loss:{losss:.4f},score:{scores:.4f},lr{optimizer.param_groups[0]['lr']:.4e}")
    #         time.update()
            if i == len(train_dataset)-1:
                print(f"epoch:{epoch},loss:{losss:.4f},score:{scores:.4f},lr{optimizer.param_groups[0]['lr']:.4e}")
            
            # 释放内存
            del loss, pred
        
    #     # 关闭进度条
    #     time.close()
        
        # 模型评估阶段
        model.eval()
        
    #     # 创建进度条以显示验证进度
    #     time = tqdm(range(len(val_dataset)))

        val_losss = 0
        val_scores = 0
        best_score = 0
        
        # 遍历验证数据集
        for i, (x, y) in enumerate(val_dataset):
            x = x.cuda().to(tc.float32)
            y = y.cuda().to(tc.float32)
            
            # 数据预处理
            x = norm_with_clip(x.reshape(-1, *x.shape[2:])).reshape(x.shape)
            
            # 使用自动混合精度进行前向传播和损失计算，但不进行梯度计算
            with autocast():
                with tc.no_grad():
                    pred = model(x)
                    loss = loss_fc(pred, y)
            
            # 计算并更新平均损失和分数
            score = dice_coef(pred.detach(), y)
            val_losss = (val_losss * i + loss.item()) / (i + 1)
            val_scores = (val_scores * i + score) / (i + 1)
            
    #         # 更新进度条
    #         time.set_description(f"val-->loss:{val_losss:.4f},score:{val_scores:.4f}")
    #         time.update()
        print(f"val-->loss:{val_losss:.4f},score:{val_scores:.4f}")

        if val_scores > best_score:
            best_score = val_scores
            # 保存模型参数
            # tc.save(model.module.state_dict(), f"./{CFG.backbone}_{epoch}_loss{losss:.2f}_score{scores:.2f}_val_loss{val_losss:.2f}_val_score{val_scores:.2f}.pt")
            tc.save(model.module.state_dict(), "./best.pt")

    #     # 关闭进度条
    #     time.close()

    
    # # 关闭最后一个进度条
    # time.close()
