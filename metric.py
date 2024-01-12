# 导入必要的库
import torch as tc  # 导入PyTorch库并重命名为tc
import torch.nn as nn  # 导入PyTorch的神经网络模块并重命名为nn
import numpy as np  # 导入NumPy库并重命名为np
from tqdm import tqdm  # 导入tqdm库用于显示循环进度条
from torch.cuda.amp import autocast  # 导入PyTorch的混合精度训练工具autocast
import cv2  # 导入OpenCV库并重命名为cv2
import os,sys  # 导入用于操作操作系统的库
from glob import glob  # 导入glob库用于文件路径匹配
import matplotlib.pyplot as plt  # 导入Matplotlib库用于绘图
import pandas as pd  # 导入Pandas库并重命名为pd
import segmentation_models_pytorch as smp  # 导入分割模型PyTorch库并重命名为smp
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据集和数据加载工具
from torch.nn.parallel import DataParallel  # 导入PyTorch的数据并行工具DataParallel


import copy

# model_path_i = 9  # 模型路径索引，选择要使用的预训练模型路径
model_path_i = 0 

class CFG:
    # ============== 模型配置 =============
    model_name = 'Unet'  # 模型名称
    backbone = 'se_resnext50_32x4d'  # 使用的骨干网络

    in_chans = 5  # 输入通道数
    
    image_size = 512 # 1024  # 输入图像大小
    input_size = 512 # 1024  # 模型输入大小
    
    tile_size = image_size  # 切片大小
    stride = tile_size // 4  # 切片步长
    drop_egde_pixel = 0  # 边缘像素丢弃数量

    target_size = 1  # 目标的通道数
    chopping_percentile = 1e-3  # 切片的百分比

    # ============== 折数 =============
    valid_id = 1  # 验证折数
    batch = 16  # 批量大小
    th_percentile = 0.00149  # 阈值百分比

#     axis_w = [0.3353333, 0.3323333, 0.3323333]  # 轴权重
    
    # my code
    axis_w = [1]

    # 预训练模型路径列表
#     model_path=["/kaggle/input/2-5d-cutting-model-baseline-training/se_resnext50_32x4d_19_loss0.12_score0.79_val_loss0.25_val_score0.79.pt",
#                "/kaggle/input/training-6-512/se_resnext50_32x4d_19_loss0.09_score0.83_val_loss0.28_val_score0.83.pt",
#                "/kaggle/input/training-6-512/se_resnext50_32x4d_19_loss0.05_score0.90_val_loss0.25_val_score0.86.pt",
#                "/kaggle/input/training-6-512/se_resnext50_32x4d_19_loss0.05_score0.89_val_loss0.24_val_score0.86_midd.pt",
#                "/kaggle/input/training-6-512/se_resnext50_32x4d_24_loss0.05_score0.90_val_loss0.23_val_score0.88_midd.pt",
#                "/kaggle/input/training-6-512/se_resnext50_32x4d_24_loss0.04_score0.91_val_loss0.23_val_score0.88_midd.pt", # 25 025 rot 512 center
#                "/kaggle/input/blood-vessel-model-1024/se_resnext50_32x4d_24_loss0.10_score0.90_val_loss0.16_val_score0.85_midd_1024.pt",
#                "/kaggle/input/blood-vessel-model-1024/se_resnext50_32x4d_24_loss0.10_score0.90_val_loss0.12_val_score0.88_midd_1024.pt",# lr = 8e-5
#                "/kaggle/input/blood-vessel-model-1024/se_resnext50_32x4d_24_loss0.91_score0.09_val_loss0.91_val_score0.09_midd_1024.pt", #  60e-5 
#                "/kaggle/input/sn-hoa-8e-5-27-rot0-5/se_resnext50_32x4d_26_loss0.10_score0.90_val_loss0.12_val_score0.88_midd_1024.pt",#8e-5-27-rot0-5
#                "/kaggle/input/sn-hoa-8e-5-27-rot0-5/se_resnext50_32x4d_30_loss0.10_score0.90_val_loss0.13_val_score0.88_midd_1024.pt"]#  31 8e 05  

    model_path=[
        "/root/xy/se_resnext50_32x4d_19_loss0.17_score0.83_val_loss0.10_val_score0.86.pt",
    ]



class CustomModel(nn.Module):
    def __init__(self, CFG, weight=None):
        super().__init__()
        
        # 初始化模型
        self.CFG = CFG
        self.model = smp.Unet(
            encoder_name=CFG.backbone, 
            encoder_weights=weight,
            in_channels=CFG.in_chans,
            classes=CFG.target_size,
            activation=None,
        )
        self.batch = CFG.batch

    def forward_(self, image):
        # 单次前向传播
        output = self.model(image)
        
        # 我感觉将(batchsize, classes=1, h, w)->(batchsize, h, w)
#         return output[:, 0]
    
        # 输出要和输入shape保持一致
        return output

    def forward(self, x: tc.Tensor):
        # x.shape=(batch, c, h, w), [80, 5, 512, 512]
        # 转换为float32类型
        x = x.to(tc.float32)
        
        # 对输入数据进行归一化处理
        # 必须要有，否则模型无法推理
        x = norm_with_clip(x.reshape(-1, *x.shape[2:])).reshape(x.shape)
        
#         # 若输入尺寸不同，则进行双线性插值调整
#         # input shape must=[b, c, h, w]
#         if CFG.input_size != CFG.image_size:
#             x = nn.functional.interpolate(x, size=(CFG.input_size, CFG.input_size), mode='bilinear', align_corners=True)
        
        # 这个shape暂存的是原图片尺寸
        shape = x.shape
    
#         # my code
#         x = nn.functional.interpolate(x, size=(CFG.input_size, CFG.input_size), mode='bilinear', align_corners=False)
#         # 使用缩放方法将图片调整到目标尺寸
#         resized_image = image.resize((CFG.input_size, CFG.input_size), Image.ANTIALIAS)

        
        # 4个图片shape=[b, c, h, w]，这里裁剪是正方形，所以怎么旋转尺寸都不变
        x_flips = 1
        x = [tc.rot90(x, k=i, dims=(-2, -1)) for i in range(x_flips)]  # 将输入进行四次旋转
        x = tc.cat(x, dim=0)
      
        
        with autocast():
            with tc.no_grad():
                # forward_(x[0:b]), forward_(x[b:2b]), ...
                # +1是为了末尾如果剩了一点样本，就也放入模型进行预测
                # res_x = [(b, 1, h, w), ...]
                # x = [self.forward_(x[i * self.batch: (i + 1) * self.batch]) for i in range(x.shape[0] // self.batch + 1)]
                x = [self.forward_(x[i * shape[0]: (i + 1) * shape[0]]) for i in range(x_flips)]

                # x shape=(4, h, w)
                x = tc.cat(x, dim=0)

                # # my code
                # x = self.forward_(x)
        
        x = x.sigmoid()  # 对输出进行sigmoid激活
        
        # 将x分为4个一组（因为之前通过旋转，一张图变成4张图）
        # shape[0]是原图片batchsize
        x = x.reshape(x_flips, shape[0], *shape[2:])

        # x:  torch.Size([4, 80, 512, 512])
        x = [tc.rot90(x[i], k=-i, dims=(-2, -1)) for i in range(x_flips)]  # 将结果进行逆时针旋转回正方向

    
        # [x_flips, 80, 512, 512]->[80, 512, 512]
        x = tc.stack(x, dim=0).mean(0)  # 取四个方向的平均值

  
        # my code
        x = tc.unsqueeze(x, dim=1) # [80, 1, 512, 512]
        # print("x: ", x.shape)
        
#         # 放缩回原图片大小
#         if CFG.input_size != CFG.image_size:
#             x = nn.functional.interpolate(x[None], size=(CFG.image_size, CFG.image_size), mode='bilinear', align_corners=True)[0]
        
#         # my code
          # 有病的规则，forward返回结果尺寸必须和forward接受结果一致
#         x = nn.functional.interpolate(x[None], size=shape, mode='bilinear', align_corners=True)[0]   
        
        # [40, 1, 512, 512]
        return x


def build_model(weight=None):

    print('model_name', CFG.model_name)
    print('backbone', CFG.backbone)

    model = CustomModel(CFG, weight)

    return model.cuda()



def min_max_normalization(x:tc.Tensor)->tc.Tensor:
    """input.shape=(batch,f1,...)"""
    shape=x.shape
    if x.ndim>2:
        x=x.reshape(x.shape[0],-1)
    
    min_=x.min(dim=-1,keepdim=True)[0]
    max_=x.max(dim=-1,keepdim=True)[0]
    if min_.mean()==0 and max_.mean()==1:
        return x.reshape(shape)
    
    x=(x-min_)/(max_-min_+1e-9)
    return x.reshape(shape)

def norm_with_clip(x:tc.Tensor,smooth=1e-5):
    dim=list(range(1,x.ndim))
    mean=x.mean(dim=dim,keepdim=True)
    std=x.std(dim=dim,keepdim=True)
    x=(x-mean)/(std+smooth)
    
    x[x>5]=(x[x>5]-5)*1e-3 +5
    x[x<-3]=(x[x<-3]+3)*1e-3-3
    return x

class Data_loader(Dataset):
    """ 
    just to load pic to memory 
    
    args:
        - path: self.paths=[pic1_path, pic2_path, ...]
     
    """
    def __init__(self,path,s="/images/"):
        # self.paths = [".../0000.tif", ".../0001.tif", ...]
        self.paths=glob(path+f"{s}*.tif")
        self.paths.sort()
        self.bool = s=="/labels/"
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,index):
        img=cv2.imread(self.paths[index],cv2.IMREAD_GRAYSCALE)
        
#         # 感觉不需要，因为模型的forward有了插值法
#         img = to_1024_1024(img , image_size = CFG.image_size )
        
        img=tc.from_numpy(img.copy())
        
        if self.bool:
            img=img.to(tc.bool)
        else:
            img=img.to(tc.uint8)
        return img

def load_data(path,s):
    """
    load memory, and remove outliers
    
    args:
        - path
        - s: is "/images/"
    
    return:
        - x: x.shape=(count, h, w)
    """
    data_loader=Data_loader(path,s)
    data_loader=DataLoader(data_loader, batch_size=16, num_workers=2)
    data=[]
    print("load the test data to computer memory! batch_size=16")
    for x in tqdm(data_loader):
        data.append(x)
    x=tc.cat(data,dim=0)

#     TH=x.reshape(-1).numpy()
#     index = -int(len(TH) * CFG.chopping_percentile)
#     TH:int = np.partition(TH, index)[index]
#     x[x>TH]=int(TH)

#     TH=x.reshape(-1).numpy()
#     index = -int(len(TH) * CFG.chopping_percentile)
#     TH:int = np.partition(TH, -index)[-index]
#     x[x<TH]=int(TH)

#     x=(min_max_normalization(x.to(tc.float16))*255).to(tc.uint8)
    return x

class Pipeline_Dataset(Dataset):
    """ 一次取self.in_chan张图做一个样本,即2.5D """
    
    def __init__(self, x, path):
        self.img_paths  = glob(path+"/images/*")
        self.img_paths.sort()
        self.in_chan = CFG.in_chans
        
        # 为了防止肾最后一张图片不能被切出in_chan//2，所以添加的边缘图片
        # 给x增加了一个全为0的通道
        # 取 x.shape 的第一个元素之外的所有元素
#         z = tc.zeros(self.in_chan//2, *x.shape[1:], dtype=x.dtype)
#         # 将张量 z、x 和 z 沿着维度 0 进行拼接
#         (amount_cut, h, w) -> (in_chan//2 + amount_cut + in_chan//2, h, w)
        
        # my code
        # .unsqueeze(0)保证x[0]和x是相同shape
        x_len = len(x)
        for i in range(self.in_chan//2):
            x = tc.cat((x[0].unsqueeze(0),x), dim=0)
            x = tc.cat((x,x[x_len-1].unsqueeze(0)), dim=0)
        self.x = x
            
        
    def __len__(self):
        """ x.shape[0]=3, self.in_chan=2, return 3-2+1=2 """
        
        return self.x.shape[0]-self.in_chan+1
    
    def __getitem__(self, index):
        """ return (self.in_chan, h, w) and it's index """
        
        x  = self.x[index:index+self.in_chan]
        return x,index
    
    def get_mark(self,index):
        """ from index to kidney_5_0000 """
        
        # /kaggle/input/blood-vessel-segmentation/test/kidney_5/images/0000.tif -> ['kidney_5', 'images', '0000.tif']
        id=self.img_paths[index].split("/")[-3:]
        
        # ['kidney_5', 'images', '0000.tif'] -> ['kidney_5', '0000.tif']
        id.pop(1)
        
        # ['kidney_5', '0000.tif'] -> 'kidney_5_0000.tif'
        id="_".join(id)
        
        # 'kidney_5_0000.tif' -> 'kidney_5_0000'
        return id[:-4]
    
    def get_marks(self):
        """ [kidney_5_0000, ...] responding 3D cuts"""
        
        ids=[]
        for index in range(len(self)):
            ids.append(self.get_mark(index))
        return ids

def add_edge(x:tc.Tensor,edge:int):
    """ x=(C,H,W), retrun (C,H+2*edge,W+2*edge) """

    mean_=int(x.to(tc.float32).mean())
    x=tc.cat([x,tc.ones([x.shape[0],edge,x.shape[2]],dtype=x.dtype,device=x.device)*mean_],dim=1)
    x=tc.cat([x,tc.ones([x.shape[0],x.shape[1],edge],dtype=x.dtype,device=x.device)*mean_],dim=2)
    x=tc.cat([tc.ones([x.shape[0],edge,x.shape[2]],dtype=x.dtype,device=x.device)*mean_,x],dim=1)
    x=tc.cat([tc.ones([x.shape[0],x.shape[1],edge],dtype=x.dtype,device=x.device)*mean_,x],dim=2)
    return x


def get_output(debug=False):
    """
    将每个肾做一个单独的数据集, 以不同轴切, 变3个肾 
    对于每个肾中的每张图, 先给其增加边缘像素, 以滑动窗口方式切, 变多张子图 
    将多张子图送入模型, 预测出多张mask, 然后拼成一张mask, 重叠部分取均值,最后再对预测mask去边缘
    最后求出每个肾的每个切片labels_列表
    
    return:
        - outputs[0]: 一个列表, 包含多个列表, 每个列表是经过TTA后求得每个肾所有切图的 labels_
        - outputs[1]: 一个列表, 依次包含每个肾的切片名字
    """
    
    
    
    # 存储输出结果的列表
    # 第三个存当前肾数据集切片的shape
    outputs = [[], [], []]

    # 如果处于调试模式，则指定一个路径；否则，获取所有测试路径
    if debug:
        paths = [
#             "/kaggle/input/blood-vessel-segmentation/train/kidney_2",
            # "/root/autodl-tmp/train/kidney_3_sparse",
#             "/kaggle/input/blood-vessel-segmentation/train/kidney_1_dense",
#             "/kaggle/input/blood-vessel-segmentation/train/kidney_1_voi",
            "/root/autodl-tmp/train/kidney_3_dense",
#             "/kaggle/input/blood-vessel-segmentation/test/kidney_5/"
        ]      
        
    else:
        # path = [".../kidney_5", ".../kidney_6"]
        paths = glob("/kaggle/input/blood-vessel-segmentation/test/*")

    # 遍历所有路径
    for path in paths:
        print(path)
        # x.shape=(count, h, w)
        x = load_data(path, "/images/") 
        
        # 创建与输入数据相同形状的零张量标签, labels.shape=(count, h, w), no lables in testing.
        labels = tc.zeros_like(x, dtype=tc.uint8)
        
        # [kidney_5_0000, ...] of x
        dataset_1kidney = Pipeline_Dataset(x, path)
        mark = dataset_1kidney.get_marks()
        
        # my code
        # in_chanels时，最少要3个样本，因为2个样本就会被3个空白样本所投输
        x = x[0:10]
        mark = mark[0:10]
        labels = tc.zeros_like(x, dtype=tc.uint8)

        # 在三个轴上进行切片，不费内存，只改变索引方式
#         for axis in [0, 1, 2]:
        for axis in [0]: # my code
            debug_count = 0

            if axis == 0:
                x_ = x
                # lables_ 是 lables 占用同一块内存,但索引的表达方式不同 
                labels_ = labels # no lables in testing, just for the same shape
            elif axis == 1:
                x_ = x.permute(1, 2, 0)
                labels_ = labels.permute(1, 2, 0)
            elif axis == 2:
                x_ = x.permute(2, 0, 1)
                labels_ = labels.permute(2, 0, 1)

            # 如果输入数据通道数为3且轴不为0，则跳出循环
            # 如果测试集只有3个样本, 则break第2，3趟循环 
            if x.shape[0] == 3 and axis != 0:
                break


            dataset = Pipeline_Dataset(x_, path)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
            
            # 获取数据集的形状
            shape = dataset.x.shape[-2:]

#             # 计算切片的坐标范围
#             x1_list = np.arange(0, shape[0] + CFG.tile_size - CFG.tile_size + 1, CFG.stride)
#             y1_list = np.arange(0, shape[1] + CFG.tile_size - CFG.tile_size + 1, CFG.stride)
            
#             # my code, to avoid errors in cutting
#             min_wh = min(shape[0], shape[1])
#             if min_wh < CFG.tile_size:
#                 # make sure that it can be divided by 32
#                 CFG.tile_size = min_wh // 32 * 32
    
    
            # my code
            # 计算切片开始切的坐标
            # stride=1, tile_size=2, shape[0]=3, => final index=3-2=1, =>arrange(0, 1+1=2)
            # x,y are the bad name
            x1_list = np.arange(0, shape[0] - CFG.tile_size , CFG.stride)
            y1_list = np.arange(0, shape[1] - CFG.tile_size , CFG.stride)

            
            # 可能因为有新增的边缘
#             x1_list = np.arange(0, shape[0] + 1, CFG.stride) # 虽然最后切不全，但还是多了很多个数据
#             y1_list = np.arange(0, shape[1] + 1, CFG.stride)
            
            
            
            print("start the inference!")
            for img, index in tqdm(dataloader):
#                 # img.shape:  torch.Size([1, 5, 1041, 1511])
#                 print("img.shape ", img.shape)

                # 将图像移动到GPU
                img = img.to("cuda:0")
            
                
#                 # 在图像边缘添加像素, 我感觉[None]表示拷贝一份新的内存给img
#                 img = add_edge(img[0], CFG.tile_size // 2)[None]

                # 初始化预测掩码及其计数
                # 选择第2维度上，第0个元素，其余维度不变
                # [1, 5, 1041, 1511] -> [1, 1041, 1511]
                mask_pred = tc.zeros_like(img[:, 0], dtype=tc.float32, device=img.device)
                mask_count = tc.zeros_like(img[:, 0], dtype=tc.float32, device=img.device)

                indexs = []
                chip = []
                # 遍历切片坐标范围
                for y1 in y1_list:
                    for x1 in x1_list:
                        
                        x2 = x1 + CFG.tile_size
                        y2 = y1 + CFG.tile_size
                        
                        # CFG.drop_egde_pixel=0
                        # indexs=[[bottom_y, top_y, left_x, left_y], ...]
                        # 记录每个切片所在整体的位置
                        indexs.append([x1 + CFG.drop_egde_pixel, x2 - CFG.drop_egde_pixel,
                                       y1 + CFG.drop_egde_pixel, y2 - CFG.drop_egde_pixel])

                        
                        # 因为测试图片太大,所以把一张图切成几份,输入到模型中
                        chip.append(img[..., x1:x2, y1:y2])

                # 我猜 y_preds.shape=[cut_counts, in_channel, h, w]
                # 将切片传递给模型进行预测

                # tc.cat(chip).shape = torch.Size([80, 5, 512, 512])
                # y_preds.shape = torch.Size([40, 1, 512, 512])
                y_preds = model.forward(tc.cat(chip)).to(device=0)

#                 # my code
#                 y_preds = model.forward(img.to(device=0))

#                 # 如果指定了边缘像素数，则在预测中去掉边缘像素
#                 if CFG.drop_egde_pixel:
#                     y_preds = y_preds[..., CFG.drop_egde_pixel:-CFG.drop_egde_pixel,
#                                         CFG.drop_egde_pixel:-CFG.drop_egde_pixel]

                # 遍历预测结果并更新掩码及其计数
#                 for i, (x1, x2, y1, y2) in enumerate(indexs):
#                     mask_pred[..., x1:x2, y1:y2] += y_preds[i]
#                     mask_count[..., x1:x2, y1:y2] += 1
                    
                # my code
                for i, (x1, x2, y1, y2) in enumerate(indexs):
                    mask_pred[..., x1:x2, y1:y2] += y_preds[i][0]
                    mask_count[..., x1:x2, y1:y2] += 1    
                
                # 某些区域重合会多加,这里计算每个像素被计算了多少次,求均值
                # mask_pred.shape = [1, 1041, 1511]
                # 感觉是整数除法，有点投票的感觉
                mask_pred /= mask_count 

#                 # 恢复预测掩码的边缘像素
#                 # tile_size // 2 是之前增加的边缘像素
#                 mask_pred = mask_pred[..., CFG.tile_size // 2:-CFG.tile_size // 2,
#                                        CFG.tile_size // 2:-CFG.tile_size // 2]

#                 # 更新标签
#                 # 预测的只是每个像素的概率，最后TH貌似就是255
#                 labels_[index] += (mask_pred[0] * 255 * CFG.axis_w[axis]).to(tc.uint8).cpu()

                # my code 
                # 取消阈值
                labels_[index] += (mask_pred[0] > 0.4).to(tc.uint8).cpu()

#                 # 如果处于调试模式，则显示图像及预测掩码
#                 # 明明img[0, CFG.in_chans // 2].shape = mask_pred[0].shape，图显示就是不一样大，气死了
#                 print("img.shape: ", img.shape)
#                 print("img.shape: ", mask_pred.shape)
#                 plt.subplot(121)
#                 plt.imshow(img[0, CFG.in_chans // 2].cpu().detach().numpy())
#                 plt.subplot(122)
#                 plt.imshow(mask_pred[0].cpu().detach().numpy())
#                 plt.show()
                    

    
        # 将标签和标记添加到输出列表
        outputs[0].append(labels)
        outputs[1].extend(mark)
        
        # my code, 对每mask都对应其原本的shape
        for i in range(len(mark)):
            outputs[2].append(list(dataset_1kidney.x.shape[-2:]))
    
    # 返回最终的输出结果
    return outputs

# 编码方式老是被metric函数中的覆盖，所以定义在这里
def rle_encode(mask):
    pixel = mask.flatten()
    pixel = np.concatenate([[0], pixel, [0]])
    run = np.where(pixel[1:] != pixel[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    if rle == '':
        rle = '1 0'
    return rle

if __name__=='__main__':

    model=build_model()

    model.load_state_dict(tc.load(CFG.model_path[ model_path_i ],"cpu"))
    model.eval()
    model=DataParallel(model)



    # 检查是否要提交，如果测试集图片数量不等于3，is_submit为True
    is_submit = len(glob("/root/autodl-tmp/test/kidney_5/images/*.tif")) != 3

    # is_submit=True # 手动规定提交模式

    # 对列表解包, 进行推理
    output, ids, shapes = get_output(not is_submit)



    # # 计算阈值
    # TH = [x.flatten().numpy() for x in output]
    # TH = np.concatenate(TH)
    # index = -int(len(TH) * CFG.th_percentile)
    # TH:int = np.partition(TH, index)[index]
    # print("TH: ",TH) # 一张图片时是Th=255

    # 初始化提交数据框
    submission_df = []
    debug_count = 0
    for index in range(len(ids)):
    
        id = ids[index] # id是一个某个切片按顺序的下标
        
        # 确定当前输出所属的肾
        i = 0
        for x in output:
            if index >= len(x):
                index -= len(x)
                i += 1
            else:
                break
        
    #     # 获取预测的血管掩码
    #     # output[i][index]: 即某张切片的预测mask
    #     mask_pred = (output[i][index] >= TH).numpy()

        # my code
        mask_pred = output[i][index].numpy()
        
        # 将掩码转换回原始大小
        # 这里可否用上采样或下采样呢
    #     mask_pred2 = to_original(mask_pred, shapes[index], image_size = CFG.image_size)
    #     mask_pred = mask_pred2.copy()

    #     # 如果不是提交模式，显示预测结果图像
    #     debug_count = 0
    #     if not is_submit:
    #         plt.subplot(131)
    # #         print(mask_pred)
    # #         plt.imshow(mask_pred)
    #         plt.imshow(output[i][index])
        
    #         pic_path = f"/kaggle/input/blood-vessel-segmentation/train/kidney_3_sparse/images/0{600+debug_count}.tif"
    #         print(pic_path)
    #         img = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
    #         plt.subplot(132)
    #         plt.imshow(img)
            
    #         pic_path2 = f"/kaggle/input/blood-vessel-segmentation/train/kidney_3_sparse/labels/0{600+debug_count}.tif"
    #         print(pic_path2)
    #         img_mask = cv2.imread(pic_path2, cv2.IMREAD_GRAYSCALE)
    #         plt.subplot(133)
    #         plt.imshow(img_mask)
            
    #         plt.show()
    #         debug_count += 1
    #         if debug_count > 6:
    #             break

        # 使用运行长度编码（Run-Length Encoding）对预测结果进行编码
        rle = rle_encode(mask_pred)

        # 将结果添加到提交数据框中
        submission_df.append(
            pd.DataFrame(data={
                'id': id,
                'rle': rle,
            }, index=[0])
        )
        

    # 合并提交数据框并保存为CSV文件
    submission_df = pd.concat(submission_df)
    submission_df.to_csv('submission.csv', index=False)

    gt_df = pd.read_csv("/root/xy/gt.csv")

    
    # to align labels 
    _gt_df = pd.merge(gt_df, submission_df.loc[:, ["id"]], on="id").reset_index(drop=True)
    from metric_lib import score
    val_score = score(_gt_df, submission_df, "id", "rle", 0) # 2D

    print(val_score)




















