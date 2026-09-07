"""
训练脚本 V2 - Cherry Branch Segmentation
=========================================

V2版本变更：
- 移除深度监督（节省显存）
- 使用高效clDice损失
- 简化训练流程

功能：
1. AMP (自动混合精度训练)
2. 余弦退火学习率调度
3. 早停机制
4. 验证集评估 (Dice Score, IoU)
5. TensorBoard日志记录

作者：Cherry Branch Segmentation Project V2
"""

import os
import argparse
import time
import json
from datetime import datetime
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import create_dataloaders
from model import CSNet, count_parameters
from loss import HybridLoss, BCEDiceLoss


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.0001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """训练器类 V2"""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        self.epochs = config.get('epochs', 100)
        self.use_amp = config.get('use_amp', True)
        
        self.scaler = GradScaler() if self.use_amp else None
        
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stop_patience', 15),
            min_delta=config.get('early_stop_min_delta', 0.0001),
            mode='min'
        )
        
        self.output_dir = config.get('output_dir', './output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        self.best_val_iou = 0.0
        self.best_dice_epoch = 0
        self.best_iou_epoch = 0
        self.best_loss_epoch = 0
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {}
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs} [Train]', leave=False)
        
        for images, masks, _ in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss, loss_dict = self.criterion(outputs, masks)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss, loss_dict = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}
        
        metrics = {'loss': avg_loss}
        metrics.update(avg_components)
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = len(self.val_loader)
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.epochs} [Val]', leave=False)
        
        for images, masks, _ in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss, _ = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss, _ = self.criterion(outputs, masks)
            
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs)
            preds_binary = (preds > 0.5).float()
            
            dice = self._compute_dice(preds_binary, masks)
            iou = self._compute_iou(preds_binary, masks)
            
            total_dice += dice
            total_iou += iou
            
            pbar.set_postfix({'loss': loss.item(), 'dice': dice, 'iou': iou})
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_iou = total_iou / num_batches
        
        return {'loss': avg_loss, 'dice': avg_dice, 'iou': avg_iou}
    
    def _compute_dice(self, pred, target, smooth=1e-5):
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return dice.item()
    
    def _compute_iou(self, pred, target, smooth=1e-5):
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
    
    def save_checkpoint(self, checkpoint_type: str, epoch: int, metric_value: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_dice': self.best_val_dice,
            'best_val_iou': self.best_val_iou,
            'best_dice_epoch': self.best_dice_epoch,
            'best_iou_epoch': self.best_iou_epoch,
            'best_loss_epoch': self.best_loss_epoch,
            'history': self.history,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if checkpoint_type == 'dice':
            best_path = os.path.join(self.output_dir, 'best_dice_model.pth')
            print(f"  ★ 保存最佳Dice模型 (Dice: {metric_value:.4f})")
        elif checkpoint_type == 'iou':
            best_path = os.path.join(self.output_dir, 'best_iou_model.pth')
            print(f"  ★ 保存最佳IoU模型 (IoU: {metric_value:.4f})")
        elif checkpoint_type == 'loss':
            best_path = os.path.join(self.output_dir, 'best_loss_model.pth')
            print(f"  ★ 保存最佳Loss模型 (Loss: {metric_value:.4f})")
        
        torch.save(checkpoint, best_path)
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*60)
        print("开始训练 V2")
        print("="*60)
        print(f"设备: {self.device}")
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"验证样本: {len(self.val_loader.dataset)}")
        print(f"批次大小: {self.train_loader.batch_size}")
        print(f"总Epochs: {self.epochs}")
        print(f"AMP: {self.use_amp}")
        print("="*60 + "\n")
        
        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['learning_rate'].append(current_lr)
            
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Dice/val', val_metrics['dice'], epoch)
            self.writer.add_scalar('IoU/val', val_metrics['iou'], epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            
            for key, value in train_metrics.items():
                if key != 'loss':
                    self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            epoch_time = time.time() - epoch_start_time
            
            save_dice = val_metrics['dice'] > self.best_val_dice
            save_iou = val_metrics['iou'] > self.best_val_iou
            save_loss = val_metrics['loss'] < self.best_val_loss
            
            if save_dice:
                self.best_val_dice = val_metrics['dice']
                self.best_dice_epoch = epoch
                self.save_checkpoint('dice', epoch, val_metrics['dice'])
            
            if save_iou:
                self.best_val_iou = val_metrics['iou']
                self.best_iou_epoch = epoch
                self.save_checkpoint('iou', epoch, val_metrics['iou'])
            
            if save_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_loss_epoch = epoch
                self.save_checkpoint('loss', epoch, val_metrics['loss'])
            
            print(f"Epoch {epoch}/{self.epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
            
            if self.early_stopping(val_metrics['loss'], epoch):
                print(f"\n早停触发!")
                break
        
        self.writer.close()
        
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\n" + "="*60)
        print("训练完成!")
        print("="*60)
        print(f"最佳Dice: {self.best_val_dice:.4f} (Epoch {self.best_dice_epoch})")
        print(f"最佳IoU:  {self.best_val_iou:.4f} (Epoch {self.best_iou_epoch})")
        print(f"最佳Loss: {self.best_val_loss:.4f} (Epoch {self.best_loss_epoch})")
        print(f"模型保存于: {self.output_dir}")
        print("  - best_dice_model.pth")
        print("  - best_iou_model.pth")
        print("  - best_loss_model.pth")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Cherry Branch Segmentation Training V2')
    
    parser.add_argument('--train-json', type=str,
                       default='/root/autodl-tmp/cherry/mmdet_data/annotations/train_roi_black_bg.json')
    parser.add_argument('--train-img', type=str,
                       default='/root/autodl-tmp/cherry/mmdet_data/train_roi_black_bg')
    parser.add_argument('--val-json', type=str,
                       default='/root/autodl-tmp/cherry/mmdet_data/annotations/test_roi_black_bg.json')
    parser.add_argument('--val-img', type=str,
                       default='/root/autodl-tmp/cherry/mmdet_data/test_roi_black_bg')
    parser.add_argument('--output-dir', type=str,
                       default='/root/autodl-tmp/cherry/branch_seg_csnet_v2/output')
    
    parser.add_argument('--loss-type', type=str, default='hybrid',
                       choices=['hybrid', 'bce_dice'])
    parser.add_argument('--bce-weight', type=float, default=1.0)
    parser.add_argument('--dice-weight', type=float, default=1.0)
    parser.add_argument('--cldice-weight', type=float, default=0.5)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--target-size', type=int, default=1024)
    
    parser.add_argument('--early-stop-patience', type=int, default=15)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-amp', action='store_true')
    
    args = parser.parse_args()
    config = vars(args)
    
    print("\n" + "="*60)
    print("Cherry Branch Segmentation V2 - Training Configuration")
    print("="*60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n加载数据集...")
    train_loader, val_loader = create_dataloaders(
        train_json=args.train_json,
        train_img_dir=args.train_img,
        val_json=args.val_json,
        val_img_dir=args.val_img,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=args.target_size
    )
    
    print("\n创建模型...")
    model = CSNet(in_channels=3, n_classes=1)
    model = model.to(device)
    print(f"参数量: {count_parameters(model):,}")
    
    print("\n创建损失函数...")
    if args.loss_type == 'hybrid':
        criterion = HybridLoss(
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            cldice_weight=args.cldice_weight,
            skeleton_iterations=3
        )
    else:
        criterion = BCEDiceLoss(
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight
        )
    
    print("\n创建优化器和调度器...")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
