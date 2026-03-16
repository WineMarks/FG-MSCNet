import os
import argparse
import time
import shutil
import logging
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
cudnn.benchmark = True

from layer.fg_mscnet import FG_MSCNet
from loss import MultiScaleCompositeLoss
from dataset import DocTamperDataset  # 确保 dataset.py 中有这个类

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        
        dist.barrier()
        return rank, local_rank, world_size
    else:
        print("Not using distributed mode")
        return 0, 0, 1

def cleanup_distributed():
    dist.destroy_process_group()

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def setup_experiment(args, rank):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(args.runs_dir, f"{timestamp}_{args.exp_name}")
    
    log_dir = os.path.join(exp_dir, 'logs')
    tb_dir = os.path.join(exp_dir, 'tensorboard')
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')

    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        logger = logging.getLogger("FG_MSCNet")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        fh = logging.FileHandler(os.path.join(log_dir, 'train.log'), mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        logger.info(f"Experiment Initialized: {args.exp_name}")
        logger.info(f"Root Dir: {exp_dir}")
        
        writer = SummaryWriter(tb_dir)
    else:
        logger = None
        writer = None

    dist.barrier()
    
    return exp_dir, ckpt_dir, logger, writer

def parse_args():
    parser = argparse.ArgumentParser(description="DDP Training for FG-MSCNet")
    parser.add_argument('--exp_name', type=str, default='icdar_03', help='Experiment Name')
    parser.add_argument('--data_dir', type=str, default='./DocTamperV1', help='Dataset Root')
    parser.add_argument('--runs_dir', type=str, default='./runs', help='Root for all experiments')
    
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size PER GPU')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=512)
    
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None, help='for finetune')
    parser.add_argument('--train_ratio', type=float, default=1.0, help='ratio of training set')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    rank, local_rank, world_size = setup_distributed()
    
    exp_dir, ckpt_dir, logger, writer = setup_experiment(args, rank)
    
    if rank == 0:
        logger.info(f"Distributed Mode: World Size = {world_size}")

    # train_dataset = DocTamperDataset(args.data_dir, mode='train', img_size=args.img_size)
    train_dataset_full = DocTamperDataset(args.data_dir, mode='train', img_size=args.img_size)

    if args.train_ratio < 1.0:
        train_size = int(len(train_dataset_full) * args.train_ratio)
        np.random.seed(42)
        train_indices = np.random.choice(len(train_dataset_full), train_size, replace=False)
        train_dataset = Subset(train_dataset_full, train_indices)
    else:
        train_dataset = train_dataset_full

    val_dataset_full = DocTamperDataset(args.data_dir, mode='val', img_size=args.img_size)

    val_ratio = 0.1
    val_size = len(val_dataset_full)
    num_val_samples = int(val_size * val_ratio)

    np.random.seed(42) 
    subset_indices = np.random.choice(val_size, num_val_samples, replace=False)
    val_dataset = Subset(val_dataset_full, subset_indices)

    if rank == 0:
        logger.info(f"📦 Train set size: {len(train_dataset)}")
        logger.info(f"📦 Val set size: {len(val_dataset)} (Reduced from {val_size} to speed up!)")
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              sampler=train_sampler, num_workers=args.num_workers, 
                              pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            sampler=val_sampler, num_workers=args.num_workers, 
                            pin_memory=True, persistent_workers=True)

    model = FG_MSCNet(num_classes=1, img_size=args.img_size).to(local_rank)

    model = model.to(memory_format=torch.channels_last)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    criterion = MultiScaleCompositeLoss().to(local_rank)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)
        else:
            progress = float(epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    best_f1 = 0.0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location={'cuda:%d' % 0: 'cuda:%d' % local_rank})
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        if rank == 0:
            logger.info(f"🔄 Resumed from epoch {start_epoch}")

    elif args.finetune and os.path.isfile(args.finetune):
        checkpoint = torch.load(args.finetune, map_location={'cuda:%d' % 0: 'cuda:%d' % local_rank})
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=False) # strict=False 容忍少许维度变动
        if rank == 0:
            logger.info(f"🚀 Loaded pretrained weights from {args.finetune} for FINE-TUNING!")

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        else:
            pbar = train_loader
            
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(local_rank, non_blocking=True, memory_format=torch.channels_last)
            masks = masks.to(local_rank, non_blocking=True)
            
            optimizer.zero_grad()

            with autocast(dtype=torch.bfloat16):
                outputs = model(images)
                loss, loss_dict = criterion(outputs, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            
            reduced_loss = reduce_tensor(loss.data, world_size)
            epoch_loss += reduced_loss.item()
            
            if rank == 0:
                pbar.set_postfix({'loss': f"{reduced_loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Train/Batch_Loss', reduced_loss.item(), global_step)

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        if rank == 0:
            logger.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
            writer.add_scalar('Train/Epoch_Loss', avg_train_loss, epoch)

        # --- Validation ---
        model.eval()
        val_f1_sum = torch.zeros(1).to(local_rank)
        val_iou_sum = torch.zeros(1).to(local_rank)
        val_p_sum = torch.zeros(1).to(local_rank)
        val_r_sum = torch.zeros(1).to(local_rank)
        val_loss_sum = torch.zeros(1).to(local_rank)
        num_batches = 0
        
        viz_input, viz_mask = None, None
        viz_preds = {}

        if rank == 0:
            logger.info("Validating...")
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        else:
            val_pbar = val_loader

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_pbar):
                images = images.to(local_rank, non_blocking=True, memory_format=torch.channels_last)
                masks = masks.to(local_rank, non_blocking=True)
                
                outputs = model(images) # outputs = [final, fine, mid, coarse]
                loss, _ = criterion(outputs, masks)
                
                val_loss_sum += loss.item()
                
                pred_prob = torch.sigmoid(outputs[0])
                pred_binary = (pred_prob > 0.5).float()

                # pred = (torch.sigmoid(outputs[0]) > 0.5).float()

                pred_np = pred_binary.cpu().numpy().flatten().astype(np.uint8)
                mask_np = masks.cpu().numpy().flatten().astype(np.uint8)
                
                f1 = f1_score(mask_np, pred_np, zero_division=1)
                iou = jaccard_score(mask_np, pred_np, zero_division=1)
                p = precision_score(mask_np, pred_np, zero_division=1)
                r = recall_score(mask_np, pred_np, zero_division=1)
                
                val_f1_sum += f1
                val_iou_sum += iou
                val_p_sum += p
                val_r_sum += r
                num_batches += 1
                
                # if rank == 0 and batch_idx == 0:
                #     viz_input = images
                #     viz_mask = masks
                #     viz_pred = pred
                if rank == 0 and batch_idx == 0:
                    viz_input = images
                    viz_mask = masks
                    viz_preds['Final'] = pred_binary
                    viz_preds['Fine_H2'] = torch.sigmoid(outputs[1])
                    viz_preds['Mid_H4'] = torch.sigmoid(outputs[2])
                    viz_preds['Coarse_H8'] = torch.sigmoid(outputs[3])
                    
                if rank == 0:
                    val_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'P': f"{p:.4f}", 'R': f"{r:.4f}", 'F1': f"{f1:.4f}"})

        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_f1_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_iou_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_p_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_r_sum, op=dist.ReduceOp.SUM)
        
        total_batches = num_batches * world_size
        avg_val_loss = val_loss_sum.item() / total_batches
        avg_val_f1 = val_f1_sum.item() / total_batches
        avg_val_iou = val_iou_sum.item() / total_batches
        avg_val_p = val_p_sum.item() / total_batches
        avg_val_r = val_r_sum.item() / total_batches
        
        if rank == 0:
            logger.info(f"Validation - Loss: {avg_val_loss:.4f} | P: {avg_val_p:.4f} | R: {avg_val_r:.4f} | F1: {avg_val_f1:.4f} | IoU: {avg_val_iou:.4f}")
            writer.add_scalar('Val/Loss', avg_val_loss, epoch)
            writer.add_scalar('Val/Precision', avg_val_p, epoch)
            writer.add_scalar('Val/Recall', avg_val_r, epoch)
            writer.add_scalar('Val/F1', avg_val_f1, epoch)
            writer.add_scalar('Val/IoU', avg_val_iou, epoch)
            state = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': max(best_f1, avg_val_f1)
            }
            
            torch.save(state, os.path.join(ckpt_dir, 'last_model.pth'))
            
            if avg_val_f1 > best_f1:
                best_f1 = avg_val_f1
                logger.info(f"New Best Model! F1: {best_f1:.4f}")
                torch.save(state, os.path.join(ckpt_dir, 'best_model.pth'))
                
            if viz_input is not None:
                target_size = viz_mask.shape[2:] 
                writer.add_images('Viz/1_Input', viz_input, epoch)
                writer.add_images('Viz/2_GT', viz_mask, epoch)
                writer.add_images('Viz/3_Pred_Final', viz_preds['Final'], epoch)
                fine_up = F.interpolate(viz_preds['Fine_H2'], size=target_size, mode='bilinear', align_corners=False)
                mid_up = F.interpolate(viz_preds['Mid_H4'], size=target_size, mode='bilinear', align_corners=False)
                coarse_up = F.interpolate(viz_preds['Coarse_H8'], size=target_size, mode='bilinear', align_corners=False)
                
                writer.add_images('Viz/4_Pred_Fine(H_2)', fine_up, epoch)
                writer.add_images('Viz/5_Pred_Mid(H_4)', mid_up, epoch)
                writer.add_images('Viz/6_Pred_Coarse(H_8)', coarse_up, epoch)

        dist.barrier()
        torch.cuda.empty_cache()

    cleanup_distributed()
    if rank == 0:
        logger.info("Training Finished.")

if __name__ == "__main__":
    main()