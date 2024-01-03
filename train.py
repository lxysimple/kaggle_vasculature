import os
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from glob import glob
import gc
import time
from collections import defaultdict
import  matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import copy
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.cuda import amp
import torch.optim as optim
import albumentations as A
import segmentation_models_pytorch as smp
import torch.nn.functional as F 
import torchvision.transforms as T


from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL


############################################################################################################
class CFG:
    inference     = False
    seed          = 42 
    
    # debug         = True
    debug         = False # a bit of data for training when it=Ture

    
    exp_name      = 'baseline'
    output_dir    = './'
    model_name    = 'Unet'
    
    backbone      = 'se_resnext50_32x4d'
    
    train_bs      = 16
    valid_bs      = 32
    img_size      = [512, 512]
    
    epochs        = 40
    
    n_accumulate  = max(1, 64//train_bs)
    lr            = 2e-3
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
#     T_max         = int(2279/(train_bs*n_accumulate)*epochs)+50
#     T_max         = int(6000/(train_bs*n_accumulate)*epochs)+50 # 6000 = 16(epoch) * 370(len(dataloader))
    T_max         = int(7000/(train_bs*n_accumulate)*epochs)+50
    
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    
    num_classes   = 1
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Kaggle
    # gt_df = "/kaggle/input/sennet-hoa-gt-data/gt.csv"
    # data_root = "/kaggle/input"
    # Linux
    gt_df = "/home/xyli/kaggle/blood-vessel-segmentation/gt.csv"
    data_root = "/home/xyli/kaggle"

    train_groups = ["kidney_1_dense", "kidney_1_voi", "kidney_2", "kidney_3_sparse"]
    valid_groups = ["kidney_3_dense"]
    
    loss_func     = "DiceLoss"

    data_transforms = {
        "train": A.Compose([
            A.Resize(*img_size, interpolation=cv2.INTER_NEAREST),
            
            # 翻转
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # 颜色变换, occupying large time, so you can open it at the end.
            A.OneOf([ # 30% to chose one of it
                A.RandomContrast(),
                A.RandomGamma(),
                A.RandomBrightness(),
                A.ColorJitter(brightness=0.07, contrast=0.07,
                              saturation=0.1, hue=0.1, always_apply=False, p=0.3)
            ], p=0.3),
            
            # 形变, occupying large time, so you can open it at the end.
            A.OneOf([ # 30% to chose one of it
                A.ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120*0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.3),
            A.ShiftScaleRotate()    
        ], p=1.0),
        
        "valid": A.Compose([
            A.Resize(*img_size, interpolation=cv2.INTER_NEAREST),
        ], p=1.0),
        
        # "valid": A.Compose([
        #     T.ToPILImage(),
        #     T.Resize(*img_size),
        #     T.ToTensor(),
        #     T.Normalize([0.625, 0.448, 0.688],
        #                 [0.131, 0.177, 0.101]),
        # ]),
        
    }

############################################################################################################

def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.'''
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    
    # When running on the CuDNN backend, two further options must be set
    # It can impact cuda performance, and you should cancel it at the end of optimization
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

############################################################################################################

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.tile(img[...,None], [1, 1, 3]) # gray to rgb
    img = img.astype('float32') # original is uint16
    mx = np.max(img)
    if mx:
        img/=mx # scale image to [0, 1]
    return img

def load_msk(path):
    msk = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    msk = msk.astype('float32')
    msk/=255.0
    return msk

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, msk_paths=[], transforms=None):
        self.img_paths  = img_paths
        self.msk_paths  = msk_paths
        self.transforms = transforms
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path  = self.img_paths[index]
        img = load_img(img_path)
        
        if len(self.msk_paths)>0:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img  = data['image']
                msk  = data['mask']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            orig_size = img.shape
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img), torch.tensor(np.array([orig_size[0], orig_size[1]]))


############################################################################################################

def build_model(backbone, num_classes, device):
    model = smp.Unet(
        encoder_name=backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#         encoder_weights="imagenet",  
        encoder_weights=None,
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_classes,        # model output channels (number of classes in your dataset)
        activation=None,
    )
    # GPUs On Server
    model = nn.DataParallel(model) 

    model.to(device)
    return model

############################################################################################################


DiceLoss = smp.losses.DiceLoss(mode='binary')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
def criterion(y_pred, y_true):  

    if CFG.loss_func == "DiceLoss":
        return DiceLoss(y_pred, y_true)
    elif CFG.loss_func == "BCELoss":
        y_true = y_true.unsqueeze(1)
        return BCELoss(y_pred, y_true)

############################################################################################################

def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.unsqueeze(1).to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.unsqueeze(1).to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

############################################################################################################
def fetch_scheduler(optimizer):
    if CFG.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG.T_max, 
                                                   eta_min=CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CFG.T_0, 
                                                             eta_min=CFG.min_lr)
    elif CFG.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=CFG.min_lr,)
    elif CFG.scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif CFG.scheduler == None:
        return None
        
    return scheduler

############################################################################################################

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, masks) in pbar: 
            
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        with amp.autocast(enabled=True):
            y_pred = model(images)
            
            loss   = criterion(y_pred, masks)

            
            loss   = loss / CFG.n_accumulate
            
        scaler.scale(loss).backward()
    
        if (step + 1) % CFG.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix( epoch=f'{epoch}',
                          train_loss=f'{epoch_loss:0.4f}',
                          lr=f'{current_lr:0.5f}',
                          gpu_mem=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    gc.collect()
    return epoch_loss

############################################################################################################


@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    val_scores = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (images, masks) in pbar:  
            
        images  = images.to(device, dtype=torch.float)
        masks   = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        y_pred  = model(images)
        loss    = criterion(y_pred, masks)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        y_pred = nn.Sigmoid()(y_pred)
        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_memory=f'{mem:0.2f} GB')
    val_scores  = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()
    return epoch_loss, val_scores

############################################################################################################

def run_training(model, optimizer, scheduler, device, num_epochs, train_loader, valid_loader):    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss      = np.inf
    best_epoch     = -1
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CFG.device, epoch=epoch)
        
        val_loss, val_scores = valid_one_epoch(model, valid_loader, 
                                                 device=CFG.device, 
                                                 epoch=epoch)
        val_dice, val_jaccard = val_scores
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)        
        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')
        print(f'Valid Loss: {val_loss}')
        
        # deep copy the model
        if val_loss <= best_loss:
            print(f"{c_}Valid loss Improved ({best_loss} ---> {val_loss})")
            best_dice    = val_dice
            best_jaccard = val_jaccard
            best_loss = val_loss
            best_epoch   = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            
            PATH = f"./{CFG.backbone}_epoch{epoch}_traloss{train_loss:.2f}_valoss{val_loss:.2f}_val_score{sum(val_scores)/len(val_scores):.2f}_best.pt"
            torch.save(model.state_dict(), PATH)
            print(f"Model Saved{sr_}")
            
        # last_model_wts = copy.deepcopy(model.state_dict())
        # PATH = PATH = f"./{CFG.backbone}_epoch{epoch}_traloss{train_loss:.2f}_valoss{val_loss:.2f}_val_score{sum(val_scores)/len(val_scores):.2f}_last.pt"
        # torch.save(model.state_dict(), PATH)
            
        print(); print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_loss))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

############################################################################################################

if __name__=='__main__':
    set_seed(CFG.seed)


    train_groups = CFG.train_groups
    valid_groups = CFG.valid_groups

    gt_df = pd.read_csv(CFG.gt_df)

    gt_df["img_path"] = gt_df["img_path"].apply(lambda x: os.path.join(CFG.data_root, x))
    gt_df["msk_path"] = gt_df["msk_path"].apply(lambda x: os.path.join(CFG.data_root, x))

    train_df = gt_df.query("group in @train_groups").reset_index(drop=True)
    valid_df = gt_df.query("group in @valid_groups").reset_index(drop=True)


    train_img_path = train_df["img_path"].values.tolist()
    train_msk_path = train_df["msk_path"].values.tolist()
    valid_img_path = valid_df["img_path"].values.tolist()
    valid_msk_path = valid_df["msk_path"].values.tolist()


    if CFG.debug:
        train_img_path = train_img_path[:CFG.train_bs*5]
        train_msk_path = train_msk_path[:CFG.train_bs*5]
        valid_img_path = valid_img_path[:CFG.valid_bs*3]
        valid_msk_path = valid_msk_path[:CFG.valid_bs*3]

    train_dataset = BuildDataset(train_img_path, train_msk_path, transforms=CFG.data_transforms['train'])
    valid_dataset = BuildDataset(valid_img_path, valid_msk_path, transforms=CFG.data_transforms['valid'])


    # The dataloader will be smaller because of batchsize.
    # num_workers=0, just to use the 0 cpu kernel.
    # train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=0, shuffle=True, pin_memory=True, drop_last=False)
    # valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=0, shuffle=False, pin_memory=True)
    
    # Use GPUs
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=2, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=2, shuffle=False, pin_memory=True)


    model = build_model(CFG.backbone, CFG.num_classes, CFG.device)



    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    scheduler = fetch_scheduler(optimizer)   


    """
    Epoch 1/15 Valid Dice: 0.3731 | Valid Jaccard: 0.2747, the beigin 
    Epoch 1/15 Valid Dice: 0.4765 | Valid Jaccard: 0.3853, efficientnet-b1->se_resnext50_32x4d

    """

    historys = []

    model, history = run_training(  
                                    model, 
                                    optimizer, 
                                    scheduler,
                                    device=CFG.device,
                                    num_epochs=CFG.epochs,
                                    train_loader=train_loader, 
                                    valid_loader=valid_loader,
                                )
    historys.append(history)















