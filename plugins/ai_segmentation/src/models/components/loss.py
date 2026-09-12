import torch
import torch.nn as nn


class Dice(nn.Module):
	def __init__(self, smooth=1.0):
		super(Dice, self).__init__()
		self.smooth = smooth

	def forward(self, y_pred, y_true):
		intersection = torch.sum(y_true * y_pred)
		union = torch.sum(y_true**2) + torch.sum(y_pred**2)
        
		return 1 - (2. * intersection + self.smooth) / (union + self.smooth)



class FocalTverskyLoss(nn.Module):
	def __init__(self, smooth=1.0, gamma=0.75, alpha=0.25):
		super(FocalTverskyLoss, self).__init__()
		self.smooth = smooth
		self.gamma = gamma
		self.alpha = alpha

	def forward(self, y_pred_batch, y_true_batch, borders):
		score = torch.tensor(0.0001, device=y_pred_batch.device).float()

		for idx, (y_pred, y_true) in enumerate(zip(y_pred_batch, y_true_batch)):
			z0, z1 = borders[0][0][idx], borders[0][1][idx]
			y0, y1 = borders[1][0][idx], borders[1][1][idx]
			x0, x1 = borders[2][0][idx], borders[2][1][idx]

			y_pred = torch.sigmoid(y_pred[0][z0:z1, y0:y1, x0:x1])
			y_pred = torch.clip(y_pred, 0.001, 0.999)
			y_true = y_true[0][z0:z1, y0:y1, x0:x1]

			true_pos = torch.sum(y_true * torch.where(y_pred > 0.5, 1.0, y_pred))
			false_neg = torch.sum(y_true * (1.0 - torch.where(y_pred > 0.5, 1.0, y_pred)))
			false_pos = torch.sum((1.0 - y_true) * torch.where(y_pred <= 0.5, 0, y_pred))

			pt_1 = (true_pos + self.smooth) / (
					true_pos + self.alpha * false_neg + (1.0 - self.alpha) * false_pos + self.smooth
			)
			score += pow(abs(1.0001 - pt_1), self.gamma)

		score /= y_pred_batch.shape[0]
		return score


class MSE_Loss(nn.Module):
	def __init__(self):
		super().__init__()
		self.mse = nn.MSELoss()

	def forward(self, y_pred, y_true, borders=None):
		return self.mse(torch.sigmoid(y_pred), y_true.float())