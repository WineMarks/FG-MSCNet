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
        pred = torch.sigmoid(pred) if not ((pred >= 0) & (pred <= 1)).all() else pred
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)

        # Dice Coeff = 2 * (A n B) / (A + B)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class EdgeLoss(nn.Module):

    def __init__(self):
        super(EdgeLoss, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        # pred: Logits -> Sigmoid
        pred = torch.sigmoid(pred)
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        pred_grad = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-6)
        with torch.no_grad():
            target_grad_x = F.conv2d(target.float(), self.sobel_x, padding=1)
            target_grad_y = F.conv2d(target.float(), self.sobel_y, padding=1)
            target_grad = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-6)
        return F.l1_loss(pred_grad, target_grad)


class MultiScaleCompositeLoss(nn.Module):
    def __init__(self, weights=(1.0, 0.75, 0.5, 0.25)):
        super(MultiScaleCompositeLoss, self).__init__()
        self.weights = weights

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.edge = EdgeLoss()
        self.lambda_bce = 1.0
        self.lambda_dice = 1.0
        self.lambda_edge = 0.5
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
        for i, pred in enumerate(preds):
            if i >= len(self.weights): break

            weight = self.weights[i]
            if pred.shape[2:] != target.shape[2:]:
                scale_factor = target.shape[2] // pred.shape[2]
                current_target = F.max_pool2d(target, kernel_size=scale_factor, stride=scale_factor)
            else:
                current_target = target
            l_bce = self.bce(pred, current_target)
            l_dice = self.dice(pred, current_target)
            l_edge = self.edge(pred, current_target)

            scale_loss = (self.lambda_bce * l_bce +
                          self.lambda_dice * l_dice +
                          self.lambda_edge * l_edge)

            total_loss += weight * scale_loss
            loss_dict["loss_total"] += weight * scale_loss.item()
            loss_dict["loss_bce"] += weight * l_bce.item()
            loss_dict["loss_dice"] += weight * l_dice.item()
            loss_dict["loss_edge"] += weight * l_edge.item()
            loss_dict[f"loss_scale_{i}"] = scale_loss.item()

        return total_loss, loss_dict


if __name__ == "__main__":
    preds = [
        torch.randn(2, 1, 384, 384),
        torch.randn(2, 1, 192, 192),
        torch.randn(2, 1, 96, 96),
        torch.randn(2, 1, 48, 48)
    ]
    target = torch.randint(0, 2, (2, 1, 384, 384)).float()
    criterion = MultiScaleCompositeLoss()
    loss, loss_dict = criterion(preds, target)

    print(f"Total Loss (Tensor): {loss}")
    print("Loss Dict details:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")