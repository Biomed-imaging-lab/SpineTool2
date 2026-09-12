import torch
import torch.nn as nn

class ProjectedDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def dice_coef(self, y_pred, y_true):
        y_pred_f = y_pred.reshape(y_pred.size(0), -1)
        y_true_f = y_true.reshape(y_true.size(0), -1)
        
        intersection = torch.sum(y_true_f * y_pred_f, dim=1)
        union = torch.sum(y_true_f**2, dim=1) + torch.sum(y_pred_f**2, dim=1)
        
        return 1 - (2. * intersection + self.smooth) / (union + self.smooth)

    def forward(self, y_pred_batch, y_true_batch, borders=None):
        # y_pred_batch: [Batch, 1, D, H, W] (Logits)
        # y_true_batch: [Batch, 1, D, H, W]
        
        # 1. Основной объемный Dice (3D)
        loss_3d = self.dice_coef(y_pred_batch, y_true_batch)
        
        # 2. Проекции (Max Projection)
        # dim=2 -> Z (Depth), dim=3 -> Y (Height), dim=4 -> X (Width)
        # В тензоре [B, C, D, H, W] оси: 0, 1, 2, 3, 4
        
        # Проекция на X (сплющиваем по X -> dim 4) -> плоскость YZ (или DH)
        # В формуле y_x - проекция по оси x.
        
        # Max pool по осям
        proj_x_pred = torch.amax(y_pred_batch, dim=4) 
        proj_x_true = torch.amax(y_true_batch, dim=4)
        
        proj_y_pred = torch.amax(y_pred_batch, dim=3)
        proj_y_true = torch.amax(y_true_batch, dim=3)
        
        proj_z_pred = torch.amax(y_pred_batch, dim=2)
        proj_z_true = torch.amax(y_true_batch, dim=2)
        
        loss_x = self.dice_coef(proj_x_pred, proj_x_true)
        loss_y = self.dice_coef(proj_y_pred, proj_y_true)
        loss_z = self.dice_coef(proj_z_pred, proj_z_true)
        
        # Итоговая сумма по формуле (3.4)
        # Усредняем по батчу
        total_loss = (0.7 * loss_3d) + (0.1 * loss_x) + (0.1 * loss_y) + (0.1 * loss_z)
        
        return total_loss.mean()
    
class SkeletonRecallLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true_skel):
        # y_pred: [Batch, 1, D, H, W] - вероятности
        # y_true_skel: [Batch, 1, D, H, W] - бинарный скелет
        intersection = torch.sum(y_pred * y_true_skel, dim=(1, 2, 3, 4))
        skel_sum = torch.sum(y_true_skel, dim=(1, 2, 3, 4))
        recall = (intersection + self.smooth) / (skel_sum + self.smooth)
        return (1.0 - recall).mean()    