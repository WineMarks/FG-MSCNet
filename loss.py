import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: [B, 1, H, W] (Logits or Probabilities)
        # target: [B, 1, H, W] (0 or 1)

        # 确保输入是概率值 (0~1)
        pred = torch.sigmoid(pred) if not ((pred >= 0) & (pred <= 1)).all() else pred

        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)

        # Dice Coeff = 2 * (A n B) / (A + B)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class EdgeLoss(nn.Module):
    """
    边缘一致性损失：
    通过 Sobel 算子提取 Pred 和 GT 的边缘，计算两者边缘图的 L1 距离。
    强制网络去拟合高频的篡改边界。
    """

    def __init__(self):
        super(EdgeLoss, self).__init__()
        # 定义 Sobel 核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        # pred: Logits -> Sigmoid
        pred = torch.sigmoid(pred)

        # 提取 Pred 边缘
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        pred_grad = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-6)

        # 提取 GT 边缘 (target 不需要梯度)
        with torch.no_grad():
            target_grad_x = F.conv2d(target.float(), self.sobel_x, padding=1)
            target_grad_y = F.conv2d(target.float(), self.sobel_y, padding=1)
            target_grad = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-6)

        # 计算边缘差异 (L1 Loss)
        return F.l1_loss(pred_grad, target_grad)


class MultiScaleCompositeLoss(nn.Module):
    def __init__(self, weights=(1.0, 0.75, 0.5, 0.25)):
        """
        weights: 对应 [Final, Mask_Fine, Mask_Mid, Mask_Coarse] 的权重
        """
        super(MultiScaleCompositeLoss, self).__init__()
        self.weights = weights

        # 子损失函数
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.edge = EdgeLoss()

        # 各个子项的内部权重
        self.lambda_bce = 1.0
        self.lambda_dice = 1.0
        self.lambda_edge = 0.5  # 边缘损失辅助即可，权重稍低

    def forward(self, preds, target):
        """
        preds: List [final_mask, mask_fine(H/2), mask_mid(H/4), mask_coarse(H/8)]
        target: Original GT [B, 1, H, W]
        """
        total_loss = 0
        loss_dict = {
            "loss_total": 0.0,
            "loss_bce": 0.0,
            "loss_dice": 0.0,
            "loss_edge": 0.0
        }

        # 遍历每个尺度的预测
        # preds[0] 是 Final (H, W)
        # preds[1] 是 Fine (H/2, W/2) ...
        for i, pred in enumerate(preds):
            if i >= len(self.weights): break

            weight = self.weights[i]

            # 1. 动态生成对应尺度的 GT
            # 如果尺寸不一致，使用 MaxPool 下采样 GT (保证篡改点不丢失)
            if pred.shape[2:] != target.shape[2:]:
                scale_factor = target.shape[2] // pred.shape[2]
                # kernel_size=scale, stride=scale
                current_target = F.max_pool2d(target, kernel_size=scale_factor, stride=scale_factor)
            else:
                current_target = target

            # 2. 计算各子项 Loss
            l_bce = self.bce(pred, current_target)
            l_dice = self.dice(pred, current_target)
            l_edge = self.edge(pred, current_target)

            # 3. 组合当前尺度的总 Loss
            # Composite = BCE + Dice + Edge
            scale_loss = (self.lambda_bce * l_bce +
                          self.lambda_dice * l_dice +
                          self.lambda_edge * l_edge)

            # 4. 加权累加到全局 Total Loss
            total_loss += weight * scale_loss

            # 5. 记录到字典 (方便 Tensorboard)
            # 我们记录加权后的值，或者记录原始值均可。这里记录加权后的贡献值。
            loss_dict["loss_total"] += weight * scale_loss.item()
            loss_dict["loss_bce"] += weight * l_bce.item()
            loss_dict["loss_dice"] += weight * l_dice.item()
            loss_dict["loss_edge"] += weight * l_edge.item()

            # 也可以记录每个尺度的单独 loss
            loss_dict[f"loss_scale_{i}"] = scale_loss.item()

        return total_loss, loss_dict


if __name__ == "__main__":
    # 模拟模型输出: [Final(384), Fine(192), Mid(96), Coarse(48)]
    preds = [
        torch.randn(2, 1, 384, 384),
        torch.randn(2, 1, 192, 192),
        torch.randn(2, 1, 96, 96),
        torch.randn(2, 1, 48, 48)
    ]

    # 模拟 GT: [B, 1, 384, 384]
    target = torch.randint(0, 2, (2, 1, 384, 384)).float()

    # 初始化 Loss
    criterion = MultiScaleCompositeLoss()

    # 计算
    loss, loss_dict = criterion(preds, target)

    print(f"Total Loss (Tensor): {loss}")
    print("Loss Dict details:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")