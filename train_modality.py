# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os
from thop import profile
from torchprofile import profile_macs
join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from utils import SurfaceDice
import torch.nn.init as init
import json
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.measure import label, regionprops
# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
#**********1
os.environ['MASTER_ADDR'] = '127.0.0.1'  # Use the IP address of your main node
os.environ['MASTER_PORT'] = '12347'     # Choose a suitable port numberdata_dir

def train_transforms(img_size, ori_h, ori_w):
    transforms = []
    if ori_h < img_size and ori_w < img_size:
        transforms.append(A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    else:
        transforms.append(A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST))
    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms, p=1.)
def get_boxes_from_mask(mask, box_num=1, std = 0.1, max_pixel = 5):
    """
    Args:
        mask: Mask, can be a torch.Tensor or a numpy array of binary mask.
        box_num: Number of bounding boxes, default is 1.
        std: Standard deviation of the noise, default is 0.1.
        max_pixel: Maximum noise pixel value, default is 5.
    Returns:
        noise_boxes: Bounding boxes after noise perturbation, returned as a torch.Tensor.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
    label_img = label(mask)
    regions = regionprops(label_img)

    # Iterate through all regions and get the bounding box coordinates
    boxes = [tuple(region.bbox) for region in regions]

    # If the generated number of boxes is greater than the number of categories,
    # sort them by region area and select the top n regions
    if len(boxes) >= box_num:
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
        boxes = [tuple(region.bbox) for region in sorted_regions]

    # If the generated number of boxes is less than the number of categories,
    # duplicate the existing boxes
    elif len(boxes) < box_num:
        num_duplicates = box_num - len(boxes)
        boxes += [boxes[i % len(boxes)] for i in range(num_duplicates)]

    # Perturb each bounding box with noise
    noise_boxes = []
    for box in boxes:
        y0, x0,  y1, x1  = box
        width, height = abs(x1 - x0), abs(y1 - y0)
        # Calculate the standard deviation and maximum noise value
        noise_std = min(width, height) * std
        max_noise = min(max_pixel, int(noise_std * 5))
         # Add random noise to each coordinate
        noise_x = np.random.randint(-max_noise, max_noise)
        noise_y = np.random.randint(-max_noise, max_noise)
        x0, y0 = x0 + noise_x, y0 + noise_y
        x1, y1 = x1 + noise_x, y1 + noise_y
        noise_boxes.append((x0, y0, x1, y1))
    return torch.as_tensor(noise_boxes, dtype=torch.float)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}



    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()



    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}



class PngDataset(Dataset):
    def __init__(self, data_dir,train_or_eval, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=1):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
        self.data_dir= data_dir

        if(train_or_eval):#训练
            dataset_other = json.load(open(os.path.join(data_dir, f'BraTS2021_endos_der_train.json'), "r"))
            dataset_ct = json.load(open(os.path.join('/notebooks/datasets/', f'CTSpine1K_train.json'), "r"))
        else:#验证
            dataset_other = json.load(open(os.path.join(data_dir, f'BraTS2021_endos_der_val.json'), "r"))
            dataset_ct = json.load(open(os.path.join('/notebooks/datasets/', f'CTSpine1K_val.json'), "r"))

        dataset_ct_key=list(dataset_ct.keys())
        dataset_other_key=list(dataset_other.keys())
        self.image_paths = dataset_ct_key+dataset_other_key
        dataset_ct_v=[dataset_ct[key] for key in dataset_ct_key]
        dataset_other_v=[dataset_other[key] for key in dataset_other_key]
        self.label_paths = dataset_ct_v + dataset_other_v
        
            
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            if(self.image_paths[index].find('ct')==-1):
                image = cv2.imread(os.path.join(self.data_dir,self.image_paths[index]))
            else:
                image = cv2.imread(os.path.join('/notebooks/datasets/',self.image_paths[index]))
            image = (image - self.pixel_mean) / self.pixel_std #**********
            h, w, _ = image.shape
        except:
            print(os.path.join(self.data_dir,self.image_paths[index]))

        
        
        transforms = train_transforms(self.image_size, h, w) #缩放到固定的256
    
        masks_list = []
        modality = 0
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        # mask_path = random.choices(self.label_paths[index], k=self.mask_num)
        m = self.label_paths[index][0] #**************

        
        if(m.find('ct')==-1):
            pre_mask = cv2.imread(os.path.join(self.data_dir,m), 0)
        else:
            pre_mask = cv2.imread(os.path.join('/notebooks/datasets/',m),0)
        
        
        try:
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255
        except Exception as e:
            print(os.path.join(self.data_dir,m),pre_mask.shape)
        

        augments = transforms(image=image, mask=pre_mask)
        image_tensor= augments['image'].float()
        mask_tensor = augments['mask'].long()
        if(m.find('ct')!=-1):
            modality=0
        elif(m.find('mr')!=-1):
            modality=1
        elif(m.find('endoscopy')!=-1):
            modality=2
        elif(m.find('dermoscopy')!=-1):
            modality=3
        boxes = get_boxes_from_mask(mask_tensor)
        masks_list.append(mask_tensor)
        boxes_list.append(boxes)
        

        mask = torch.stack(masks_list, dim=0)
        boxes = torch.stack(boxes_list, dim=0)
        b,_,_=boxes.shape
        # boxes = torch.squeeze(boxes)
        boxes = boxes.view(b, 4) 
        # boxes=torch.squeeze(boxes, dim=2)
       
        # image_input["image"] = 
        # image_input["label"] = mask.unsqueeze(1)
        # image_input["boxes"] = boxes
        
        image_name = self.image_paths[index].split('/')[-1]
        
        return (
            image_tensor,#.unsqueeze(0)  torch.tensor(img_1024).float(),
            mask,#.unsqueeze(1)
            modality,
            boxes,
            image_name,
        )
        
    def __len__(self):
        return len(self.image_paths)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,                                   
    default="/notebooks/datasets/other",   
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument(
    "-v",
    "--val_npy_path", 
    type=str,       
    default="/notebooks/datasets/other",  
    help="path to validation npy files; two subfolders: gts and imgs",
)                                         
parser.add_argument("-task_name", type=str, default="us_medsam")
parser.add_argument("-model_type", type=str, default="vit_b") 
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth" 
)#work_dir/MedSAM/medsam_vit_b.pth          
# parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=40)
parser.add_argument("-num_workers", type=int, default=2)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=True, help="use amp")
## Distributed training args
parser.add_argument("--world_size", type=int, help="world size")
parser.add_argument("--node_rank", type=int, default=0, help="Node rank")
parser.add_argument(
    "--bucket_cap_mb",
    type=int,
    default=25,
    help="The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)",
)
parser.add_argument(
    "--grad_acc_steps",
    type=int,
    default=2,
    help="Gradient accumulation steps before syncing gradients for backprop",
)
parser.add_argument(
    "--resume", type=str, default=""
)
parser.add_argument("--init_method", type=str, default="env://")

args = parser.parse_args()

if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )

# %% set up model for fine-tuning
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)


# %% set up model
class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        
        for name, param in image_encoder.named_parameters():
            if name.find('mlp')==-1 :
                param.requires_grad = False
        for name, param in mask_decoder.named_parameters():
            if name.find('mlp')==-1 and name.find('iou_prediction_head')==-1:
            # if name.find('gating_network')==-1 :    
                param.requires_grad = False

        # # 扩展新模态需要冻结的结构
        # for name, param in image_encoder.named_parameters(): #5
        #     if name.find('experts.3.')==-1 and name.find('gating_network')==-1:
        #         param.requires_grad = False
        # for name, param in mask_decoder.named_parameters():
        #     if name.find('experts.3.')==-1 and name.find('gating_network')==-1:
        #         param.requires_grad = False


    def forward(self, image,modality, box):
        image_embedding = self.image_encoder(image,modality)  # (B, 256, 64, 64) 
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
            modality=modality,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks#

args.init_method = "tcp://localhost:12345"  # Use the same IP and port you used for MASTER_ADDR and MASTER_PORT

def main():
    ngpus_per_node = torch.cuda.device_count()
    print("Spwaning processces")
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))



def main_worker(gpu, ngpus_per_node, args):
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size =4#args.world_size #*********************************
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")
    is_main_host = rank == 0
    if is_main_host:
        os.makedirs(model_save_path, exist_ok=True)
        shutil.copyfile(
            __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
        )
    torch.cuda.set_device(gpu)
    # device = torch.device("cuda:{}".format(gpu))
    torch.distributed.init_process_group(
        backend="nccl", init_method=args.init_method, rank=rank, world_size=world_size
    )
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)#checkpoint=args.checkpoint

    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder
    ).cuda()
    


    cuda_mem_info = torch.cuda.mem_get_info(gpu)
    free_cuda_mem, total_cuda_mem = cuda_mem_info[0] / (1024**3), cuda_mem_info[1] / (
        1024**3
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Total CUDA memory before DDP initialised: {total_cuda_mem} Gb"
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Free CUDA memory before DDP initialised: {free_cuda_mem} Gb"
    )
    if rank % ngpus_per_node == 0:
        print("Before DDP initialization:")
        os.system("nvidia-smi")

    medsam_model = nn.parallel.DistributedDataParallel(
        medsam_model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=args.bucket_cap_mb,  ## Too large -> comminitation overlap, too small -> unable to overlap with computation
    )

    cuda_mem_info = torch.cuda.mem_get_info(gpu)
    free_cuda_mem, total_cuda_mem = cuda_mem_info[0] / (1024**3), cuda_mem_info[1] / (
        1024**3
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Total CUDA memory after DDP initialised: {total_cuda_mem} Gb"
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Free CUDA memory after DDP initialised: {free_cuda_mem} Gb"
    )
    if rank % ngpus_per_node == 0:
        print("After DDP initialization:")
        os.system("nvidia-smi")


    
    # 初始化
    # ema = EMA(medsam_model, 0.9999)
    # ema.register()
    medsam_model.train()

    encoder_parms=0
    for blk in medsam_model.module.image_encoder.blocks:
        encoder_parms+=sum(p.numel() for p in blk.mlp.parameters())
    print("Number of encoder mlp parameters:",encoder_parms)

    decoder_parms=sum(p.numel() for p in medsam_model.module.mask_decoder.output_hypernetworks_mlps.parameters())+sum(p.numel() for p in medsam_model.module.mask_decoder.iou_prediction_head.parameters())
    for ll in medsam_model.module.mask_decoder.transformer.layers:
        decoder_parms+=sum(p.numel() for p in ll.mlp.parameters())
    print("Number of decoder mlp parameters:",decoder_parms)
    print("Number of mlp parameters:",encoder_parms+decoder_parms)

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    ) 
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    ) 

    ## Setting up optimiser and loss func
    # only optimize the parameters of image encodder, mask decoder, do not update prompt encoder
    # img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(medsam_model.mask_decoder.parameters())
    img_mask_encdec_params = list(
        medsam_model.module.image_encoder.parameters()
    ) + list(medsam_model.module.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  # 93729252
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    losses_eval = []
    best_loss = 1e10
    train_dataset = PngDataset(args.tr_npy_path,train_or_eval=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_dataset = PngDataset(args.val_npy_path,train_or_eval=False)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    ## Distributed sampler has done the shuffling for you,
    ## So no need to shuffle in dataloader

    print("Number of training samples: ", len(train_dataset))
    print("Number of validation samples: ", len(val_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=(val_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=val_sampler,
    )
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(rank, "=> loading checkpoint '{}'".format(args.resume))
            ## Map model to be loaded to specified single GPU
            loc = "cuda:{}".format(gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                rank,
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                ),
            )
        torch.distributed.barrier()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print(f"[RANK {rank}: GPU {gpu}] Using AMP for training")

    best_dsc=0
    best_epoch=0
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        epoch_tr_dsc=0
        #print(len(train_dataloader))
        train_dataloader.sampler.set_epoch(epoch)
        for step, (image, gt2D, modality,boxes, _) in enumerate(
            tqdm(train_dataloader, desc=f"[RANK {rank}: GPU {gpu}]")
        ):
            # print(image.shape,gt2D.shape)
            # print(gt2D[0])
            optimizer.zero_grad()
            # print(image.shape,gt2D.shape,boxes.shape)
            boxes_np = boxes.detach().cpu().numpy()
            # image, gt2D = image.to(device), gt2D.to(device)
            image, gt2D = image.cuda(), gt2D.cuda()
            modality=torch.tensor(modality).cuda()
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image,modality, boxes_np)#&&&&&&&
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                        medsam_pred, gt2D.float()
                    )
                    # loss.requires_grad = True
                    # Gradient accumulation
                    if args.grad_acc_steps > 1:
                        loss = (
                            loss / args.grad_acc_steps
                        )  # normalize the loss because it is accumulated
                        if (step + 1) % args.grad_acc_steps == 0:
                            ## Perform gradient sync
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            # ema.update()
                            optimizer.zero_grad()
                        else:
                            ## Accumulate gradient on current node without backproping
                            with medsam_model.no_sync():
                                
                                loss.backward()  ## calculate the gradient only  
                    else:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        # ema.update()
                        optimizer.zero_grad() 
                dsc=SurfaceDice.compute_dice_coefficient(gt2D.to(torch.bool),medsam_pred>0)
            else:
                medsam_pred = medsam_model(image,modality, boxes_np)#&&&&&&&&&&
                # print(medsam_pred.shape,gt2D.shape)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                    medsam_pred, gt2D.float()
                ) 
                # Gradient accumulation
                if args.grad_acc_steps > 1:
                    loss = (
                        loss / args.grad_acc_steps
                    )  # normalize the loss because it is accumulated
                    if (step + 1) % args.grad_acc_steps == 0:
                        ## Perform gradient sync
                        loss.backward()
                        optimizer.step()
                        # ema.update()
                        optimizer.zero_grad()
                    else:
                        ## Accumulate gradient on current node without backproping
                        with medsam_model.no_sync():
                            loss.backward()  ## calculate the gradient only
                else:
                    loss.backward()
                    optimizer.step()
                    # ema.update()
                    optimizer.zero_grad()
                dsc=SurfaceDice.compute_dice_coefficient(gt2D.to(torch.bool),medsam_pred>0)
  
    
            epoch_loss += loss.item()
            epoch_tr_dsc+=dsc
            iter_num += 1
        step=step+1
        epoch_loss /= step###############step+1
        epoch_tr_dsc/= step###############step+1
        losses.append(epoch_loss)
        
        # Check CUDA memory usage
        cuda_mem_info = torch.cuda.mem_get_info(gpu)
        free_cuda_mem, total_cuda_mem = cuda_mem_info[0] / (1024**3), cuda_mem_info[
            1
        ] / (1024**3)
        
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        
        print( f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Train_loss: {epoch_loss}, DSC: {epoch_tr_dsc}'
)
        print("\n")
        torch.distributed.barrier()

        # %% plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        # plt.show() # comment this line if you are running on a server
        plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()

        #validation
        # ema.apply_shadow()
        medsam_model.eval()  # 切换到评估模式，禁用梯度计算
        epoch_loss_val = 0
        epoch_dsc=0
        # 验证集的数据加载器
        # print("***************************")
        
        for step, (image, gt2D,modality, boxes, _) in enumerate(
            tqdm(val_dataloader, desc=f"[RANK {rank}: GPU {gpu}]")
        ):
            boxes_np = boxes.detach().cpu().numpy()
            # image, gt2D = image.to(device), gt2D.to(device)
            image, gt2D = image.cuda(), gt2D.cuda()
            modality=torch.tensor(modality).cuda()
            with torch.no_grad():  # 禁用梯度计算
                medsam_pred = medsam_model(image,modality, boxes_np)#
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                    medsam_pred, gt2D.float()
                )#+
            dsc=SurfaceDice.compute_dice_coefficient(gt2D.to(torch.bool),medsam_pred>0)
            epoch_dsc+=dsc    
            epoch_loss_val += loss.item()
        step=step+1
        epoch_loss_val /= step###############step+1
        epoch_dsc/= step   ###############step+1
        losses_eval.append(epoch_loss_val)
        if(epoch_dsc>best_dsc):
            best_dsc=epoch_dsc
            best_epoch=epoch
        
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Val_loss: {epoch_loss_val}, DSC: {epoch_dsc}, Best_dsc: {best_dsc}, Best_epoch: {best_epoch}'
        )
        plt.plot(losses_eval)
        plt.title("Dice + Cross Entropy Eval Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        # plt.show() # comment this line if you are running on a server
        plt.savefig(join(model_save_path, args.task_name + "_eval_loss.png"))
        plt.close()
        if(epoch>=5 and epoch<=100 and (epoch+1) % 10 == 0 ):  #第二阶段
            checkpoint = {
                    "model": medsam_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
            torch.save(checkpoint, join(model_save_path, "epoch_"+str(epoch)+".pth"))#EMA_
        # ema.restore()

if __name__ == "__main__": 
    main()
