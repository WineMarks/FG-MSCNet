import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from layer.fg_mscnet import FG_MSCNet
from dataset import DocTamperDataset 

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1

class MetricAccumulator:
    def __init__(self, thresholds, device):
        self.thresholds = thresholds
        self.num_th = len(thresholds)
        self.tp = torch.zeros(self.num_th, device=device)
        self.fp = torch.zeros(self.num_th, device=device)
        self.tn = torch.zeros(self.num_th, device=device)
        self.fn = torch.zeros(self.num_th, device=device)

    def update(self, preds, targets):
        for i, th in enumerate(self.thresholds):
            pred_b = (preds >= th).float()
            self.tp[i] += (pred_b * targets).sum()
            self.fp[i] += (pred_b * (1 - targets)).sum()
            self.fn[i] += ((1 - pred_b) * targets).sum()
            self.tn[i] += ((1 - pred_b) * (1 - targets)).sum()

    def get_metrics(self):
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-6)
        acc = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + 1e-6)
        return precision, recall, f1, iou, acc

def generate_visualizations(save_dir, dataset_name, idx, image, gt, pred, attn_map, pred_as_heatmap=False):
    base_save_path = os.path.join(save_dir, dataset_name)
    os.makedirs(base_save_path, exist_ok=True)
    
    mean = torch.tensor([0.485, 0.455, 0.406]).view(3, 1, 1).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)
    img_show = image * std + mean
    img_show = img_show.squeeze().permute(1, 2, 0).cpu().numpy()
    img_show = np.clip(img_show, 0, 1)

    gt_show = gt.squeeze().cpu().numpy()
    pred_show = pred.squeeze().cpu().numpy()
    
    error_map = np.zeros_like(img_show)
    error_map[(gt_show == 1) & (pred_show < 0.5)] = [1, 0, 0] # FN
    error_map[(gt_show == 0) & (pred_show >= 0.5)] = [0, 0, 1] # FP
    
    if pred_as_heatmap:
        pred_display = pred_show
        pred_cmap = 'jet'
    else:
        pred_display = (pred_show >= 0.5).astype(np.float32)
        pred_cmap = 'gray'

    if attn_map is not None:
        N = attn_map.shape[2]
        H = W = int(np.sqrt(N))
        attn_spatial = attn_map.squeeze().sum(dim=-1).reshape(H, W).numpy()
        attn_spatial = cv2.resize(attn_spatial, (img_show.shape[1], img_show.shape[0]))
        attn_spatial = (attn_spatial - attn_spatial.min()) / (attn_spatial.max() - attn_spatial.min() + 1e-6)
    else:
        attn_spatial = np.zeros((img_show.shape[0], img_show.shape[1]))

    plt.imsave(os.path.join(base_save_path, f"sample_{idx}_img.png"), img_show)
    plt.imsave(os.path.join(base_save_path, f"sample_{idx}_gt.png"), gt_show, cmap='gray')
    plt.imsave(os.path.join(base_save_path, f"sample_{idx}_pred.png"), pred_display, cmap=pred_cmap)
    plt.imsave(os.path.join(base_save_path, f"sample_{idx}_error.png"), error_map)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(img_show); axes[0].set_title("Image")
    axes[1].imshow(gt_show, cmap='gray'); axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_display, cmap=pred_cmap); axes[2].set_title("Prediction")
    axes[3].imshow(error_map); axes[3].set_title("Error Map (Red=FN, Blue=FP)")
    axes[4].imshow(img_show); axes[4].imshow(attn_spatial, cmap='magma', alpha=0.6); axes[4].set_title("FG-SRA Attention")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(base_save_path, f"sample_{idx}_concat.png"), dpi=150)
    plt.close()

def evaluate_dataset(args, model, dataset_name, rank, local_rank, world_size):
    dataset = DocTamperDataset(args.data_dir, mode=dataset_name, img_size=args.img_size)
    sampler = DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=2, pin_memory=True)
    
    thresholds = torch.arange(0.1, 1.0, 0.1).to(local_rank)
    tracker = MetricAccumulator(thresholds, local_rank)
    
    auc_preds, auc_targets = [], []
    
    if rank == 0:
        print(f"\nEvaluating on [{dataset_name.upper()}] ...")
        pbar = tqdm(loader, desc=f"Eval [{dataset_name.upper()}]")
    else:
        pbar = loader
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(local_rank, non_blocking=True, memory_format=torch.channels_last)
            masks = masks.to(local_rank, non_blocking=True)
            
            with autocast():
                outputs = model(images)
                pred_prob = torch.sigmoid(outputs[0])
            tracker.update(pred_prob, masks)
            
            p_flat = pred_prob.view(-1)
            t_flat = masks.view(-1)
            indices = torch.randperm(p_flat.size(0), device=local_rank)[:2000]
            auc_preds.append(p_flat[indices].cpu().numpy())
            auc_targets.append(t_flat[indices].cpu().numpy())
            if rank == 0 and batch_idx < args.save_n:
                attn = None
                if hasattr(model.module, 'enc1') and hasattr(model.module.enc1, 'attn'):
                    attn = model.module.enc1.attn.last_attn_map
                generate_visualizations(args.save_dir, dataset_name, batch_idx, 
                                        images[0], masks[0], pred_prob[0], attn, args.pred_as_heatmap)
            del outputs, pred_prob, images, masks
            if batch_idx % 10 == 0: 
                torch.cuda.empty_cache()
    dist.all_reduce(tracker.tp, op=dist.ReduceOp.SUM)
    dist.all_reduce(tracker.fp, op=dist.ReduceOp.SUM)
    dist.all_reduce(tracker.tn, op=dist.ReduceOp.SUM)
    dist.all_reduce(tracker.fn, op=dist.ReduceOp.SUM)
    
    precision, recall, f1, iou, acc = tracker.get_metrics()
    best_idx = torch.argmax(f1)
    best_th = thresholds[best_idx].item()
    local_preds = np.concatenate(auc_preds)
    local_targets = np.concatenate(auc_targets)
    auc = roc_auc_score(local_targets, local_preds) if len(np.unique(local_targets)) > 1 else 0.5
    ap = average_precision_score(local_targets, local_preds) if len(np.unique(local_targets)) > 1 else 0.0
    
    return {
        'Dataset': dataset_name,
        'Best_Th': best_th,
        'F1': f1[best_idx].item(),
        'Precision': precision[best_idx].item(),
        'Recall': recall[best_idx].item(),
        'mIoU': iou[best_idx].item(),
        'Accuracy': acc[best_idx].item(),
        'AUC-ROC': auc,
        'AP': ap
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./DocTamperV1')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--save_dir', type=str, default='./eval_results')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--save_n', type=int, default=50, help='Number of top visualizations to save')
    parser.add_argument('--pred_as_heatmap', action='store_true', help='Save prediction as color heatmap instead of binary mask')
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    model = FG_MSCNet(num_classes=1, img_size=args.img_size).to(local_rank)
    model = model.to(memory_format=torch.channels_last)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    ckpt = torch.load(args.checkpoint, map_location={'cuda:%d' % 0: 'cuda:%d' % local_rank})
    model.module.load_state_dict(ckpt['model_state_dict'])
    
    datasets = ['fcd', 'scd', 'test']
    # datasets = ['test']
    all_results = []
    
    for ds_name in datasets:
        res = evaluate_dataset(args, model, ds_name, rank, local_rank, world_size)
        if rank == 0: all_results.append(res)
        dist.barrier()
        
    # 生成 Markdown 对比表格
    if rank == 0:
        print("\n" + "="*80)
        print("🎯 Cross-Dataset Performance Audit Report")
        print("="*80)
        header = "| Dataset | Best Th | F1-Score | Precision | Recall | mIoU | Pixel Acc | AUC-ROC | AP |"
        print(header)
        print("|" + "|".join(["---"] * 9) + "|")
        for r in all_results:
            row = f"| {r['Dataset'].upper()} | {r['Best_Th']:.1f} | {r['F1']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['mIoU']:.4f} | {r['Accuracy']:.4f} | {r['AUC-ROC']:.4f} | {r['AP']:.4f} |"
            print(row)
        print("="*80)
        print(f"Visualizations saved to: {args.save_dir}")

if __name__ == "__main__":
    main()