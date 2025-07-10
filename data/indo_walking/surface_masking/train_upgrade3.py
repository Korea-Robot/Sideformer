# train.py - 수정된 버전

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
import wandb
import numpy as np

from surface_dataset import SurfaceSegmentationDataset

# 설정
metadata_path = "./processed_dataset/metadata.json"
batch_size = 32  # 배치 크기 감소 (더 안정적인 학습)
target_size = (512, 512)
num_workers = 16

# 데이터 로드
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

train_dataset = SurfaceSegmentationDataset(metadata['train_data'], target_size, is_train=True)
valid_dataset = SurfaceSegmentationDataset(metadata['valid_data'], target_size, is_train=False)

# 클래스 가중치 계산 함수
def calculate_class_weights(dataset, num_classes=7):
    """클래스 불균형을 해결하기 위한 가중치 계산"""
    class_counts = np.zeros(num_classes)
    
    print("클래스 분포 계산 중...")
    for i in range(min(len(dataset), 1000)):  # 샘플링해서 계산
        mask = dataset[i]['labels'].numpy()
        for class_idx in range(num_classes):
            class_counts[class_idx] += np.sum(mask == class_idx)
    
    # 역가중치 계산
    total_pixels = np.sum(class_counts)
    class_weights = total_pixels / (num_classes * class_counts)
    
    # 배경 클래스 가중치 조정 (너무 높지 않게)
    class_weights[0] = min(class_weights[0], 0.5)
    
    return torch.FloatTensor(class_weights)

# 클래스 매핑
class_to_idx = {
    'background': 0,
    'caution_zone': 1,
    'bike_lane': 2,
    'alley': 3,
    'roadway': 4,
    'braille_guide_blocks': 5,
    'sidewalk': 6
}

id2label = {int(idx): label for label, idx in class_to_idx.items()}
label2id = class_to_idx
num_labels = len(id2label)

print(f"클래스 수: {num_labels}")
print(f"클래스 레이블: {id2label}")

# 데이터로더 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=num_workers, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                         num_workers=num_workers, pin_memory=True)

# 클래스 가중치 계산
class_weights = calculate_class_weights(train_dataset, num_labels)
print(f"클래스 가중치: {class_weights}")

class DirectSegFormer(nn.Module):
    def __init__(self, pretrained_model_name="nvidia/mit-b0", num_classes=7):
        super().__init__()
        
        self.original_model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # 정규화 파라미터 (ImageNet 표준)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x):
        # 정규화 (입력은 0-1 범위)
        x = (x - self.mean) / self.std
        outputs = self.original_model(pixel_values=x)
        return outputs.logits

# 모델 및 학습 설정
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = DirectSegFormer(num_classes=num_labels)
model.to(device)

# 하이퍼파라미터
epochs = 100
lr = 3e-5  # 학습률 감소
weight_decay = 1e-4

# 옵티마이저
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# 클래스 가중치를 적용한 손실 함수
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# 학습률 스케줄러
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# WandB 초기화
wandb.init(project="segmentation_project", name=f"fixed_run_0707")

# 검증 함수
def validate_model(model, valid_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for val_data in valid_loader:
            val_imgs = val_data['pixel_values'].to(device)
            val_masks = val_data['labels'].to(device)
            
            val_logits = model(val_imgs)
            val_outputs = F.interpolate(
                val_logits,
                size=val_masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            loss = criterion(val_outputs, val_masks)
            val_loss += loss.item()
            
            # 정확도 계산
            preds = torch.argmax(val_outputs, dim=1)
            correct_pixels += (preds == val_masks).sum().item()
            total_pixels += val_masks.numel()
    
    avg_val_loss = val_loss / len(valid_loader)
    pixel_accuracy = correct_pixels / total_pixels
    
    return avg_val_loss, pixel_accuracy

# 학습 루프
model.train()
best_val_loss = float('inf')

for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    
    for i, data in enumerate(train_loader):
        images = data['pixel_values'].to(device)
        masks = data['labels'].to(device)
        
        # 순전파
        logits = model(images)
        upsampled_logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        # 손실 계산
        loss = criterion(upsampled_logits, masks)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    # 검증
    val_loss, pixel_acc = validate_model(model, valid_loader, criterion, device)
    
    # 학습률 스케줄러 업데이트
    scheduler.step()
    
    # 평균 손실 계산
    avg_train_loss = running_loss / len(train_loader)
    
    # WandB 로깅
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
        "pixel_accuracy": pixel_acc,
        "learning_rate": scheduler.get_last_lr()[0]
    })
    
    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Pixel Acc: {pixel_acc:.4f}")
    
    # 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"ckpts/best_seg_model.pth")
        print(f"Best 모델 저장! Val Loss: {val_loss:.4f}")
    
    # 정기적으로 체크포인트 저장
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"ckpts/seg_model_epoch_{epoch+1}.pth")

print("학습 완료!")
wandb.finish()