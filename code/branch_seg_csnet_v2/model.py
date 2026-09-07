"""
网络模型模块 - CS-Net (Curvilinear Structure Network) V2
==========================================================

专为细长曲线结构分割设计的网络架构 - V2版本

V2版本变更：
- 关闭深度监督 (节省显存，让clDice可以运行)

核心特点：
1. U-Net编码器-解码器结构
2. 空间注意力机制 (Spatial Attention) - 关注细长结构位置
3. 通道注意力机制 (Channel Attention) - 自适应特征选择
4. 多尺度特征融合 - 捕获不同粗细的枝条

作者：Cherry Branch Segmentation Project V2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    """基础卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * attention.expand_as(x)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        
        return x * attention


class CurveContextResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.dilated = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.gain = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.fuse(torch.cat([self.local(x), self.dilated(x)], dim=1))
        return x + self.gain * context


class CSAttentionBlock(nn.Module):
    """曲线结构注意力块"""
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        use_channel_attention: bool = True,
        use_spatial_attention: bool = True,
    ):
        super().__init__()
        self.channel_attention = (
            ChannelAttention(channels, reduction) if use_channel_attention else nn.Identity()
        )
        self.spatial_attention = (
            SpatialAttention(kernel_size=7) if use_spatial_attention else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class EncoderBlock(nn.Module):
    """编码器块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_channel_attention: bool = True,
        use_spatial_attention: bool = True,
        use_curve_context: bool = False,
    ):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.attention = CSAttentionBlock(
            out_channels,
            use_channel_attention=use_channel_attention,
            use_spatial_attention=use_spatial_attention,
        )
        self.curve_context = CurveContextResidual(out_channels) if use_curve_context else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x = self.attention(x)
        x = self.curve_context(x)
        pooled = self.pool(x)
        return x, pooled


class DecoderBlock(nn.Module):
    """解码器块"""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_channel_attention: bool = True,
        use_spatial_attention: bool = True,
        use_curve_context: bool = False,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)
        self.attention = CSAttentionBlock(
            out_channels,
            use_channel_attention=use_channel_attention,
            use_spatial_attention=use_spatial_attention,
        )
        self.curve_context = CurveContextResidual(out_channels) if use_curve_context else nn.Identity()
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.attention(x)
        x = self.curve_context(x)
        
        return x


class CSNet(nn.Module):
    """
    CS-Net V2: Curvilinear Structure Network
    
    V2版本：关闭深度监督，节省显存用于clDice
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 1,
        use_channel_attention: bool = True,
        use_spatial_attention: bool = True,
        use_curve_context: bool = False,
    ):
        super().__init__()

        attention_kwargs = {
            "use_channel_attention": use_channel_attention,
            "use_spatial_attention": use_spatial_attention,
        }
        self.enc1 = EncoderBlock(in_channels, 64, **attention_kwargs)
        self.enc2 = EncoderBlock(64, 128, **attention_kwargs)
        self.enc3 = EncoderBlock(128, 256, **attention_kwargs)
        self.enc4 = EncoderBlock(256, 512, **attention_kwargs)
        
        self.bottleneck = nn.Sequential(
            ConvBlock(512, 1024),
            CSAttentionBlock(1024, **attention_kwargs)
        )
        
        self.dec4 = DecoderBlock(1024, 512, 512, **attention_kwargs)
        self.dec3 = DecoderBlock(512, 256, 256, **attention_kwargs)
        self.dec2 = DecoderBlock(256, 128, 128, **attention_kwargs, use_curve_context=use_curve_context)
        self.dec1 = DecoderBlock(128, 64, 64, **attention_kwargs, use_curve_context=use_curve_context)
        
        self.output_head = nn.Conv2d(64, n_classes, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.size()[2:]
        
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        
        x = self.bottleneck(x)
        
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        output = self.output_head(x)
        
        if output.size()[2:] != input_size:
            output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        
        return output


class CSNetAux(nn.Module):
    """
    CS-Net V2 with auxiliary semantic head (training-only).

    Main head: 1ch foreground/background (same as CSNet).
    Aux head:  3ch background/trunk/branch (removed at inference).

    Backbone shared with CSNet.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        self.bottleneck = nn.Sequential(
            ConvBlock(512, 1024),
            CSAttentionBlock(1024),
        )

        self.dec4 = DecoderBlock(1024, 512, 512)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)

        # Main head: binary foreground
        self.main_head = nn.Conv2d(64, 1, 1)
        # Aux head: 3-class semantic (bg=0, trunk=1, branch=2) — training only
        self.aux_head = nn.Conv2d(64, 3, 1)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        input_size = x.size()[2:]

        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        main = self.main_head(x)
        if main.size()[2:] != input_size:
            main = F.interpolate(main, size=input_size, mode='bilinear', align_corners=True)

        if return_aux:
            aux = self.aux_head(x)
            if aux.size()[2:] != input_size:
                aux = F.interpolate(aux, size=input_size, mode='bilinear', align_corners=True)
            return main, aux

        return main


def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("="*60)
    print("CS-Net V2 模型测试")
    print("="*60)
    
    model = CSNet(in_channels=3, n_classes=1)
    model.eval()
    
    x = torch.randn(2, 3, 1024, 1024)
    
    with torch.no_grad():
        output = model(x)
        print(f"\n输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
    
    print(f"\n模型参数量: {count_parameters(model):,}")
