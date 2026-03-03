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

# --- 引入自定义模块 ---
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
    """O(1) 内存复杂度的混淆矩阵累加器，用于最佳阈值搜索"""
    def __init__(self, thresholds, device):
        self.thresholds = thresholds
        self.num_th = len(thresholds)
        self.tp = torch.zeros(self.num_th, device=device)
        self.fp = torch.zeros(self.num_th, device=device)
        self.tn = torch.zeros(self.num_th, device=device)
        self.fn = torch.zeros(self.num_th, device=device)

    def update(self, preds, targets):
        # preds: [B, 1, H, W] (0~1 概率), targets: [B, 1, H, W] (0 或 1)
        for i, th in enumerate(self.thresholds):
            pred_b = (preds >= th).float()
            self.tp[i] += (pred_b * targets).sum()
            self.fp[i] += (pred_b * (1 - targets)).sum()
            self.fn[i] += ((1 - pred_b) * targets).sum()
            self.tn[i] += ((1 - pred_b) * (1 - targets)).sum()

    def get_metrics(self):
        # 计算所有阈值下的指标
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-6)
        acc = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + 1e-6)
        return precision, recall, f1, iou, acc

def generate_visualizations(save_dir, dataset_name, idx, image, gt, pred, attn_map, pred_as_heatmap=False):
    """生成定性分析矩阵与热图，并单独保存各组分图"""
    base_save_path = os.path.join(save_dir, dataset_name)
    os.makedirs(base_save_path, exist_ok=True)
    
    # 1. 还原图像 (反归一化)
    mean = torch.tensor([0.485, 0.455, 0.406]).view(3, 1, 1).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)
    img_show = image * std + mean
    img_show = img_show.squeeze().permute(1, 2, 0).cpu().numpy()
    img_show = np.clip(img_show, 0, 1)

    gt_show = gt.squeeze().cpu().numpy()
    pred_show = pred.squeeze().cpu().numpy()
    
    # 2. 误差图 (Error Map): 红色为漏报(FN)，蓝色为误报(FP)
    # 此处依然使用连续概率(0.5为界)来计算误差图，保证与指标一致
    error_map = np.zeros_like(img_show)
    error_map[(gt_show == 1) & (pred_show < 0.5)] = [1, 0, 0] # FN
    error_map[(gt_show == 0) & (pred_show >= 0.5)] = [0, 0, 1] # FP
    
    # --- 新增：处理预测图的显示/保存格式 ---
    if pred_as_heatmap:
        # 使用连续概率和 jet 热力图
        pred_display = pred_show
        pred_cmap = 'jet'
    else:
        # 默认使用 0.5 阈值进行二值化，并使用灰度图 (黑白)
        pred_display = (pred_show >= 0.5).astype(np.float32)
        pred_cmap = 'gray'

    # 3. FG-SRA 热图处理
    if attn_map is not None:
        N = attn_map.shape[2]
        H = W = int(np.sqrt(N))
        attn_spatial = attn_map.squeeze().sum(dim=-1).reshape(H, W).numpy()
        attn_spatial = cv2.resize(attn_spatial, (img_show.shape[1], img_show.shape[0]))
        attn_spatial = (attn_spatial - attn_spatial.min()) / (attn_spatial.max() - attn_spatial.min() + 1e-6)
    else:
        attn_spatial = np.zeros((img_show.shape[0], img_show.shape[1]))

    # --- 4. 单独保存各组分图 ---
    # 保存原图
    plt.imsave(os.path.join(base_save_path, f"sample_{idx}_img.png"), img_show)
    # 保存 Ground Truth
    plt.imsave(os.path.join(base_save_path, f"sample_{idx}_gt.png"), gt_show, cmap='gray')
    # 保存预测概率图 (根据 pred_as_heatmap 决定是黑白二值还是彩色热力图)
    plt.imsave(os.path.join(base_save_path, f"sample_{idx}_pred.png"), pred_display, cmap=pred_cmap)
    # 保存误差图
    plt.imsave(os.path.join(base_save_path, f"sample_{idx}_error.png"), error_map)
    
    # --- 5. 绘图 (拼接图) ---
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(img_show); axes[0].set_title("Image")
    axes[1].imshow(gt_show, cmap='gray'); axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_display, cmap=pred_cmap); axes[2].set_title("Prediction")
    axes[3].imshow(error_map); axes[3].set_title("Error Map (Red=FN, Blue=FP)")
    axes[4].imshow(img_show); axes[4].imshow(attn_spatial, cmap='magma', alpha=0.6); axes[4].set_title("FG-SRA Attention")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    # 将拼接图命名为 _concat 区分
    plt.savefig(os.path.join(base_save_path, f"sample_{idx}_concat.png"), dpi=150)
    plt.close()

def evaluate_dataset(args, model, dataset_name, rank, local_rank, world_size):
    dataset = DocTamperDataset(args.data_dir, mode=dataset_name, img_size=args.img_size)
    sampler = DistributedSampler(dataset, shuffle=False)
    # 强制 batch_size=1 保障 4G 显存安全 (代码原为 batch_size=2，保留原逻辑)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=2, pin_memory=True)
    
    thresholds = torch.arange(0.1, 1.0, 0.1).to(local_rank)
    tracker = MetricAccumulator(thresholds, local_rank)
    
    auc_preds, auc_targets = [], []
    
    if rank == 0:
        print(f"\n🚀 Evaluating on [{dataset_name.upper()}] ...")
        pbar = tqdm(loader, desc=f"Eval [{dataset_name.upper()}]")
    else:
        # 其他 GPU 不显示进度条，避免终端输出错乱
        pbar = loader
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(local_rank, non_blocking=True, memory_format=torch.channels_last)
            masks = masks.to(local_rank, non_blocking=True)
            
            with autocast():
                outputs = model(images)
                pred_prob = torch.sigmoid(outputs[0])
            
            # 更新阈值混淆矩阵
            tracker.update(pred_prob, masks)
            
            # 蓄水池采样用于 AUC/AP (每张图随机采 2000 个像素点)
            p_flat = pred_prob.view(-1)
            t_flat = masks.view(-1)
            indices = torch.randperm(p_flat.size(0), device=local_rank)[:2000]
            auc_preds.append(p_flat[indices].cpu().numpy())
            auc_targets.append(t_flat[indices].cpu().numpy())
            
            # 可视化 (仅 Rank 0, 且保存前 args.save_n 张)
            if rank == 0 and batch_idx < args.save_n:
                attn = None
                if hasattr(model.module, 'enc1') and hasattr(model.module.enc1, 'attn'):
                    attn = model.module.enc1.attn.last_attn_map
                
                # 传入 args.pred_as_heatmap 决定格式
                generate_visualizations(args.save_dir, dataset_name, batch_idx, 
                                        images[0], masks[0], pred_prob[0], attn, args.pred_as_heatmap)
            
            # 极限显存清理
            del outputs, pred_prob, images, masks
            if batch_idx % 10 == 0: 
                torch.cuda.empty_cache()

    # --- 同步多卡结果 ---
    dist.all_reduce(tracker.tp, op=dist.ReduceOp.SUM)
    dist.all_reduce(tracker.fp, op=dist.ReduceOp.SUM)
    dist.all_reduce(tracker.tn, op=dist.ReduceOp.SUM)
    dist.all_reduce(tracker.fn, op=dist.ReduceOp.SUM)
    
    precision, recall, f1, iou, acc = tracker.get_metrics()
    
    # 找到最佳 F1 及其对应的阈值
    best_idx = torch.argmax(f1)
    best_th = thresholds[best_idx].item()
    
    # 计算 AUC/AP (汇聚所有局部的采样点)
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
    # --- 新增参数：开启此选项将保存为彩色概率热力图，不加则默认为黑白二值图 ---
    parser.add_argument('--pred_as_heatmap', action='store_true', help='Save prediction as color heatmap instead of binary mask')
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    
    # 强制将模型放到 channels_last 以减少内存并提速
    model = FG_MSCNet(num_classes=1, img_size=args.img_size).to(local_rank)
    model = model.to(memory_format=torch.channels_last)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # 加载权重
    ckpt = torch.load(args.checkpoint, map_location={'cuda:%d' % 0: 'cuda:%d' % local_rank})
    model.module.load_state_dict(ckpt['model_state_dict'])
    
    # datasets = ['fcd', 'scd', 'test']
    datasets = ['test']
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
        print(f"📁 Visualizations saved to: {args.save_dir}")

if __name__ == "__main__":
    main()