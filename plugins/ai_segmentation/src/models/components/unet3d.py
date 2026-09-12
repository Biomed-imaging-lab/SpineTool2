import torch
import torch.nn as nn

from src.models.components.lora_conv3d import LoRAConv3d, DoRAConv3d

def get_adapter_conv3d(lora_adapter):
	if lora_adapter == "dora":
		return DoRAConv3d
	if lora_adapter == "lora":
		return LoRAConv3d
	raise ValueError(f"Unknown lora_adapter: {lora_adapter}")

class Swish(nn.Module):
	def forward(self, x):
		return x * torch.sigmoid(x)


class conv_block(nn.Module):
	def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1, bias=True, lora_rank=0, lora_alpha=None, lora_adapter="lora"):
		super(conv_block, self).__init__()
		Conv3dAdapter = get_adapter_conv3d(lora_adapter)
		self.conv = nn.Sequential(
			Conv3dAdapter(in_channels, out_channels, k_size, stride, padding, bias=bias, lora_rank=lora_rank, lora_alpha=lora_alpha),
			nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.01),
			Swish(),
			Conv3dAdapter(out_channels, out_channels, k_size, stride, padding, bias=bias, lora_rank=lora_rank, lora_alpha=lora_alpha),
			nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.01),
			Swish(),
		)

	def forward(self, x):
		return self.conv(x)


class up_conv(nn.Module):
	def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1, bias=True, lora_rank=0, lora_alpha=None, lora_adapter="lora"):
		super(up_conv, self).__init__()
		Conv3dAdapter = get_adapter_conv3d(lora_adapter)
		self.up = nn.Sequential(
			nn.Upsample(scale_factor=2),
			Conv3dAdapter(in_channels, out_channels, k_size, stride, padding, bias=bias, lora_rank=lora_rank, lora_alpha=lora_alpha),
			nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.01),
			Swish(),
		)

	def forward(self, x):
		return self.up(x)


class U_Net(nn.Module):
	def __init__(self, in_ch=1, out_ch=1, init_features=32, lora_rank=0, lora_alpha=None, lora_adapter="lora"):
		super(U_Net, self).__init__()
		n1 = init_features
		filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

		self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

		self.Conv1 = conv_block(in_ch, filters[0], lora_rank=lora_rank, lora_alpha=lora_alpha, lora_adapter=lora_adapter)
		self.Conv2 = conv_block(filters[0], filters[1], lora_rank=lora_rank, lora_alpha=lora_alpha, lora_adapter=lora_adapter)
		self.Conv3 = conv_block(filters[1], filters[2], lora_rank=lora_rank, lora_alpha=lora_alpha, lora_adapter=lora_adapter)
		self.Conv4 = conv_block(filters[2], filters[3], lora_rank=lora_rank, lora_alpha=lora_alpha, lora_adapter=lora_adapter)
		self.Conv5 = conv_block(filters[3], filters[4], lora_rank=lora_rank, lora_alpha=lora_alpha, lora_adapter=lora_adapter)

		self.Up5 = up_conv(filters[4], filters[3], lora_rank=lora_rank, lora_alpha=lora_alpha, lora_adapter=lora_adapter)
		self.Up_conv5 = conv_block(filters[4], filters[3], lora_rank=lora_rank, lora_alpha=lora_alpha, lora_adapter=lora_adapter)
		self.Up4 = up_conv(filters[3], filters[2], lora_rank=lora_rank, lora_alpha=lora_alpha, lora_adapter=lora_adapter)
		self.Up_conv4 = conv_block(filters[3], filters[2], lora_rank=lora_rank, lora_alpha=lora_alpha, lora_adapter=lora_adapter)
		self.Up3 = up_conv(filters[2], filters[1], lora_rank=lora_rank, lora_alpha=lora_alpha, lora_adapter=lora_adapter)
		self.Up_conv3 = conv_block(filters[2], filters[1], lora_rank=lora_rank, lora_alpha=lora_alpha, lora_adapter=lora_adapter)
		self.Up2 = up_conv(filters[1], filters[0], lora_rank=lora_rank, lora_alpha=lora_alpha, lora_adapter=lora_adapter)
		self.Up_conv2 = conv_block(filters[1], filters[0], lora_rank=lora_rank, lora_alpha=lora_alpha, lora_adapter=lora_adapter)

		Conv3dAdapter = get_adapter_conv3d(lora_adapter)
		self.Conv = Conv3dAdapter(filters[0], out_ch, kernel_size=1, stride=1, padding=0, lora_rank=lora_rank, lora_alpha=lora_alpha)

	def forward(self, x):
		e1 = self.Conv1(x)
		e2 = self.Maxpool(e1)
		e2 = self.Conv2(e2)
		e3 = self.Maxpool(e2)
		e3 = self.Conv3(e3)
		e4 = self.Maxpool(e3)
		e4 = self.Conv4(e4)
		e5 = self.Maxpool(e4)
		e5 = self.Conv5(e5)

		d5 = self.Up5(e5)
		d5 = torch.cat((e4, d5), dim=1)
		d5 = self.Up_conv5(d5)

		d4 = self.Up4(d5)
		d4 = torch.cat((e3, d4), dim=1)
		d4 = self.Up_conv4(d4)

		d3 = self.Up3(d4)
		d3 = torch.cat((e2, d3), dim=1)
		d3 = self.Up_conv3(d3)

		d2 = self.Up2(d3)
		d2 = torch.cat((e1, d2), dim=1)
		d2 = self.Up_conv2(d2)

		out = self.Conv(d2)
		return out