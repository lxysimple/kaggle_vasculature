"""
和baseline不同之处：
    数据增强有点不一样，我去掉了一点，新增了一点自己的
    验证集数据增强不一样，baseline验证集没有裁剪图片
    去掉了add_noise
    因为数据增强有裁剪，所以去掉了随机裁剪，我感觉把一张图依次裁剪完整别随机裁，可能更好
    数据增强中，真的适合mask也增强吗，我记得smp中有关于分割的数据增强
    去掉了随机翻转、旋转手工数据增强

"""


# ============================ import libraries ============================

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

from datetime import datetime

# ============================ global configure ============================

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
    # backbone = 'se_resnext101_32x4d'

    in_chans = 5 # 1/5  # 输入通道数, 我感觉是5张图片看做一个样本

    # ============== 训练配置 =============
    # Expected image height and width divisible by 32.
    image_size = 1280 # 896/512/1024/1280  # 图片大小 
    input_size = 1280 # 896/512/1024/1280  # 输入尺寸

    # input_size=1920, in_chans=5, 1-GPU-max—memory's batch=3, 2.35G/2.45G, 95% 
    train_batch_size = 16 # 16 # 训练批量大小
    valid_batch_size = train_batch_size * 2  # 验证批量大小
    num_workers = 2

    epochs = 40 # 20/40  # 训练轮数
    
    lr = 6e-5  # 学习率

    chopping_percentile = 1e-3  # 切割百分比

    data_root = '/home/xyli/kaggle/blood-vessel-segmentation'
    # data_root = '/root/autodl-tmp/'
    # data_root = '/root/autodl-tmp'

    paths = [
                f"{data_root}/train/kidney_1_dense",
                f"{data_root}/train/kidney_1_voi",
                f"{data_root}/train/kidney_2",
                f"{data_root}/train/kidney_3_sparse",

                # f"{data_root}/train/kidney_3_dense",
            ]

    # ============== 折数 =============
    valid_id = 1  # 验证集编号

    # ============== 数据增强 =============

    # https://blog.csdn.net/zhangyuexiang123/article/details/107705311
    train_aug_list = [
        # useless
        # A.RandomGamma(p=0.75),  # 随机Gamma变换

        # my code
        # 只有当input_size很大时才开启，这样随机裁剪就失效了
        A.Resize(height=input_size, width=input_size, p=1),

        A.Rotate(limit=45, p=0.5),  # 旋转
        A.RandomScale(scale_limit=(0.8, 1.25), interpolation=cv2.INTER_CUBIC, p=0.5),  # 随机缩放

        A.RandomCrop(input_size, input_size, p=1),  # 随机裁剪

        A.RandomBrightnessContrast(p=0.5, ),  # 随机亮度对比度变换
        A.GaussianBlur(p=0.5),  # 高斯模糊
        A.MotionBlur(p=0.5),  # 运动模糊
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),  # 网格扭曲

        ToTensorV2(transpose_mask=True),  # 转换为张量
    ]
    train_aug = A.Compose(train_aug_list)
    valid_aug_list = [
        # my code
        # 只有当input_size很大时才开启，这样随机裁剪就失效了
        A.Resize(height=input_size, width=input_size, p=1),

        # 注意这个不是整张图片，而是在随机裁剪的图片上做验证的
        A.RandomCrop(input_size, input_size, p=1),  

        ToTensorV2(transpose_mask=True),  # 转换为张量
    ]
    valid_aug = A.Compose(valid_aug_list)

# ============================ the model ============================
    
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

# mv /home/xyli/kaggle/kaggle_vasculature/workplace/se_resnext50_32x4d-a260b3a4.pth /home/xyli/.cache/torch/hub/checkpoints/
# mv /home/.cache/torch/checkpoints/se_resnext101_32x4d-3b2fe3d8.pth /root/.cache/torch/hub/checkpoints/se_resnext101_32x4d-3b2fe3d8.pth
def build_model(weight="imagenet"):
    # # 加载环境变量
    # load_dotenv()

    # local_weights_path = '/home/xyli/kaggle/kaggle_vasculature/workplace/se_resnext50_32x4d-a260b3a4.pth'
    
    # 输出模型名称和使用的骨干网络
    print('model_name', CFG.model_name)
    print('backbone', CFG.backbone)

    # 构建并返回模型
    model = CustomModel(CFG, weight)

    # # my code
    # model = CustomModel(CFG, None)
    # model.load_state_dict(tc.load('/root/xy/best_loss.pt'))

    return model.cuda()

# ============================ batch normalization ============================

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

    # 对大于5和小于-3的值进行截断处理，即去除噪音
    x[x > 5] = (x[x > 5] - 5) * 1e-3 + 5
    x[x < -3] = (x[x < -3] + 3) * 1e-3 - 3

    return x

# ============================ add noise ============================

# def add_noise(x: tc.Tensor, max_randn_rate=0.1, randn_rate=None, x_already_normed=False):
#     """
#     给定输入张量 x, 添加噪声并返回处理后的张量

#     Args:
#         - x: 输入张量，形状为 (batch, f1, f2, ...).
#         - max_randn_rate: 随机噪声的最大比例，默认为 0.1.
#         - randn_rate: 可选参数, 手动指定噪声比例，如果为 None 则随机生成.
#         - x_already_normed: 布尔值, 指示输入张量是否已经进行了标准化.

#     Returns:
#         处理后的张量, 其方差已被归一化.

#     Warning:
#         - 如果输入张量已经标准化 (x_already_normed=True)，则 x_std 为全 1 张量, x_mean 为全 0 张量.
#         - 如果输入张量未标准化，根据输入的维度进行标准化处理.

#     Reference:
#         - https://blog.csdn.net/chaosir1991/article/details/106960408
#     """
#     ndim = x.ndim - 1

#     if x_already_normed:
#         x_std = tc.ones([x.shape[0]] + [1] * ndim, device=x.device, dtype=x.dtype)
#         x_mean = tc.zeros([x.shape[0]] + [1] * ndim, device=x.device, dtype=x.dtype)
#     else:
#         dim = list(range(1, x.ndim))
#         x_std = x.std(dim=dim, keepdim=True)
#         x_mean = x.mean(dim=dim, keepdim=True)

#     if randn_rate is None:
#         randn_rate = max_randn_rate * np.random.rand() * tc.rand(x_mean.shape, device=x.device, dtype=x.dtype)

#     # 计算噪声缩放系数
#     cache = (x_std ** 2 + (x_std * randn_rate) ** 2) ** 0.5 + 1e-7

#     # 添加噪声并返回处理后的张量
#     return (x - x_mean + tc.randn(size=x.shape, device=x.device, dtype=x.dtype) * randn_rate * x_std) / cache

# ============================ dataset just for loading ============================

class Data_loader(Dataset):
    """" just put the data into the memory. """

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

# ============================ the model ============================
    
def load_data(paths, is_label=False):
    """ 用空间换时间 """

    # 创建Dataset对象，处理数据路径和是否为标签的标志
    data_loader = Data_loader(paths, is_label)
    # 创建DataLoader对象，设置批量大小为16，使用2个工作进程加载数据
    data_loader = DataLoader(data_loader, batch_size=CFG.train_batch_size, num_workers=CFG.num_workers)
    # 存储数据的列表
    data = []
    # 遍历数据加载器，将每个批次的数据添加到列表中
    # for x in tqdm(data_loader):
    #     data.append(x)
    for x in data_loader: # x data/iter
        data.append(x)
    
    # 列表->张量
    x = tc.cat(data, dim=0)
    # 释放内存，删除数据列表
    del data
    
    # 如果不是标签数据
    if not is_label:

        # ============== 对数据进行百分比上截断处理,去除异常点 ===============

        # x拉成1维
        TH = x.reshape(-1).numpy()
        # 根据设定的百分比确定阈值位置
        index = -int(len(TH) * CFG.chopping_percentile)
        # 对阈值进行分区操作，并取得分区后的阈值
        TH: int = np.partition(TH, index)[index]
        # 将大于阈值的元素设置为阈值
        x[x > TH] = int(TH)

        # ============== 下截断处理 ===============

        TH = x.reshape(-1).numpy()
        # 根据设定的百分比确定阈值位置
        index = -int(len(TH) * CFG.chopping_percentile)
        # 对阈值进行分区操作，并取得分区后的阈值
        TH: int = np.partition(TH, -index)[-index]
        # 将小于阈值的元素设置为阈值
        x[x < TH] = int(TH)
        
        # ============== 归一化 ===============

        # 对数据进行最小-最大归一化，并将数据类型转换为uint8
        x = (min_max_normalization(x.to(tc.float16)[None])[0] * 255).to(tc.uint8)
    
    return x # 返回处理后的数据张量

# ============================ validation metric ============================

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

# ============================ train loss ============================

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
    
# ============================ train loss ============================
    
class Kaggld_Dataset(Dataset):
    def __init__(self, x: list, y: list, arg: bool = False):
        super(Dataset, self).__init__()
        self.x = x  
        self.y = y  # mask
        self.image_size = CFG.image_size  # 图像大小
        self.in_chans = CFG.in_chans  # 输入通道数
        self.arg = arg  # 是否进行数据增强
        if arg:
            self.transform = CFG.train_aug  # 训练数据增强配置
        else:
            self.transform = CFG.valid_aug  # 验证数据增强配置

    def __len__(self) -> int:
        # 每in_chans个样本看做一个样本，即2.5D
        return sum([y.shape[0] - self.in_chans for y in self.y])

    def __getitem__(self, index):

        # 某个数据集末尾的图片数凑不齐in_chans个的话，就从下一个数据集开始
        # 感觉这个算法有点问题，另外不如把所有数据融到一起简单易懂
        i = 0
        for x in self.x:
            if index > x.shape[0] - self.in_chans:
                index -= x.shape[0] - self.in_chans
                i += 1
            else:
                break
        x = self.x[i] # 换到下一个肾数据集
        y = self.y[i]
            
        # # 在图中裁剪(image_size, image_size)区域，x_index定义裁剪开始位置
        # # x_index = np.random.randint(0, x.shape[1] - self.image_size )
        # # y_index = np.random.randint(0, x.shape[2] - self.image_size )
        # # my code
        # x_index = np.random.randint(0, x.shape[1] - self.image_size + 1)
        # y_index = np.random.randint(0, x.shape[2] - self.image_size + 1)

        # # 同时对in_chans个图进行裁剪
        # x = x[index:index + self.in_chans, x_index:x_index + self.image_size, y_index:y_index + self.image_size]
        # # 取中间的mask做为该2.5D样本的mask
        # y = y[index + self.in_chans // 2, x_index:x_index + self.image_size, y_index:y_index + self.image_size]

        # my code
        x = x[index:index + self.in_chans, :, :]
        y = y[index + self.in_chans // 2, :, :]


        # 我感觉是为了与其他图像处理库或工具兼容，因为一些库（如 Matplotlib）期望图像的通道表示是 (H, W, C) 的形式
        data = self.transform(image=x.numpy().transpose(1, 2, 0), mask=y.numpy())
        x = data['image']
        y = data['mask'] >= 127 # ratate时会出现大于127的值，即异常值

        if self.arg:
            i = np.random.randint(4)
            # x是3维，y是2维
            x = x.rot90(i, dims=(1, 2))
            y = y.rot90(i, dims=(0, 1))
            for i in range(3):
                if np.random.randint(2):
                    x = x.flip(dims=(i,))
                    if i >= 1:
                        y = y.flip(dims=(i - 1,))
        return x, y  # 返回处理后的图像数据，类型为(uint8, uint8)

# ============================ the main ============================

if __name__=='__main__':

    # =============== optimize GPUs =============== 

    # 启用cudnn加速，并进行基准测试
    tc.backends.cudnn.enabled = True
    tc.backends.cudnn.benchmark = True
 
    # =============== data path ===============

    train_x = [] # train_x=[[all pic of kidney_1_dense], [all pic of kidney_1_voi], ...]
    train_y = [] # is the corresponding mask
    
    paths = CFG.paths 

    # 'path' is the directory path of [kidney_1_dense, kidney_1_voi, ...]
    for i, path in enumerate(paths): 

        # 排除特定路径, but I think it will not be run.
        if path == f"{CFG.data_root}/train/kidney_3_dense":    
            continue
        
        # 每次加载一个数据集，也是一个3D肾
        # 这里90%已经被自动排序了
        x = load_data(glob(f"{path}/images/*"), is_label=False)
        print("train dataset x shape:", x.shape)
        
        # 加载标签数据
        y = load_data(glob(f"{path}/labels/*"), is_label=True)
        print("train dataset y shape:", y.shape)
        
        
        # 将数据添加到训练集, (c,h,w)
        train_x.append(x)
        train_y.append(y)

        # 对1个3D肾进行特有的切片数据增强
        # 维度变换,(h,w,c),本来是以z轴切图的,现在以x轴切图
        train_x.append(x.permute(1, 2, 0))
        train_y.append(y.permute(1, 2, 0))
        # (w,c,h),以y轴切
        train_x.append(x.permute(2, 0, 1))
        train_y.append(y.permute(2, 0, 1))
    
    # 验证集路径
    path1 = f"{CFG.data_root}/train/kidney_3_sparse"
    path2 = f"{CFG.data_root}/train/kidney_3_dense"

    # 获取验证集图像和标签路径列表
    paths_y = glob(f"{path2}/labels/*")
    paths_x = [x.replace("labels", "images").replace("dense", "sparse") for x in paths_y]

    # =============== data path ===============

    # =============== load the data ===============

    # 加载验证集图像和标签数据
    val_x = load_data(paths_x, is_label=False)
    print("validate dataset x shape:", val_x.shape)
    val_y = load_data(paths_y, is_label=True)
    print("validate dataset y shape:", val_y.shape)	

    # =============== load the data ===============

    # =============== define objects ===============
    
    # train_x=[kidney1{cut by z}, kidney1{cut by x}, kidney1{cut by y}, ...]
    train_dataset = Kaggld_Dataset(train_x, train_y, arg=True)
    # 创建训练数据加载器，设置批大小、工作线程数、是否打乱数据、是否将数据存储在固定内存中
    # train_dataset = DataLoader(train_dataset, batch_size=CFG.train_batch_size, num_workers=2, shuffle=True, pin_memory=True)
    train_dataset = DataLoader(train_dataset, batch_size=CFG.train_batch_size, num_workers=CFG.num_workers, shuffle=False, pin_memory=True)

    # 创建验证数据集对象，使用Kaggld_Dataset类，传入验证数据和标签
    val_dataset = Kaggld_Dataset([val_x], [val_y])
    # 创建验证数据加载器，设置批大小、工作线程数、是否打乱数据、是否将数据存储在固定内存中
    # val_dataset = DataLoader(val_dataset, batch_size=CFG.valid_batch_size, num_workers=2, shuffle=False, pin_memory=True)
    val_dataset = DataLoader(val_dataset, batch_size=CFG.valid_batch_size, num_workers=CFG.num_workers, shuffle=False, pin_memory=True)

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

    # 使用OneCycleLR策略，单位:epoch
    # 刚开始学习率逐步增加，快速收敛；之后学习率逐步减小，进一步收敛；最后继续减少，巩固收敛
    # scheduler = tc.optim.lr_scheduler.OneCycleLR(
    #     optimizer, 
    #     max_lr=CFG.lr,
    #     steps_per_epoch=len(train_dataset), 
    #     epochs=CFG.epochs+1,
    #     pct_start=0.1
    # )

    # MultiStepLR 存在BUG
    scheduler = tc.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=20, 
        gamma=0.1,
    )

    # =============== define objects ===============

    # =============== start the train ===============

    print("start the train!")
    best_score = 0.0
    best_valid = 999.0
    for epoch in range(CFG.epochs):

        # =============== train ===============

        model.train()
               
        losss = 0
        scores = 0
        for i, (x, y) in enumerate(train_dataset):

            x = x.cuda().to(tc.float32)
            y = y.cuda().to(tc.float32)
            
            # 数据预处理
            x = norm_with_clip(x.reshape(-1, *x.shape[2:])).reshape(x.shape)
            # x = add_noise(x, max_randn_rate=0.5, x_already_normed=True) # 测试过不提分
            
            # 使用自动混合精度进行前向传播和损失计算
            with autocast():
                pred = model(x)
                loss = loss_fc(pred, y)
            
            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()
            # scheduler.step()
            
            # 计算并更新平均损失和分数
            score = dice_coef(pred.detach(), y)
            losss = (losss * i + loss.item()) / (i + 1)
            scores = (scores * i + score) / (i + 1)
            
            if i == len(train_dataset)-1:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},epoch:{epoch},loss:{losss:.4f},score:{scores:.4f},lr{optimizer.param_groups[0]['lr']:.4e}")
            
            # 释放显存
            del loss, pred

        scheduler.step() # 不同的scheduler的优化单位不一样

        # =============== validation ===============

        model.eval()
        
        val_losss = 0
        val_scores = 0
        
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
            
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},val-->loss:{val_losss:.4f},score:{val_scores:.4f}")
        print()


        if val_scores > best_score:
            best_score = val_scores
            tc.save(model.module.state_dict(), f"./{CFG.backbone}_{epoch}_loss{losss:.2f}_score{scores:.2f}_val_loss{val_losss:.2f}_val_score{val_scores:.2f}.pt")
            # tc.save(model.module.state_dict(), "./best_score.pt")
            
        if val_losss < best_valid:
            best_valid = val_losss
            tc.save(model.module.state_dict(), f"./{CFG.backbone}_{epoch}_loss{losss:.2f}_score{scores:.2f}_val_loss{val_losss:.2f}_val_score{val_scores:.2f}.pt")
            # tc.save(model.module.state_dict(), "./best_loss.pt")
    
