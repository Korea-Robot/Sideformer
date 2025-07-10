# GPU Device Setting

# pytorch나 transformer 라이브러리 보다 먼저 이것을 가으와져오야함.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"

# SegFormer 모델 학습을 위한 완전한 코드
import json
import torch
# torch.cuda.set_device(1)

from torch.utils.data import Dataset, DataLoader
from torch import nn
import evaluate
from datasets import load_dataset
from torchvision.transforms import ColorJitter
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer
)

from surface_dataset import SurfaceSegmentationDataset
# --- 설정 ---
# 전처리된 데이터 정보가 담긴 metadata.json 파일 경로
metadata_path = "./processed_dataset/metadata.json"

# 학습 파라미터
batch_size = 64
target_size = (512, 512)
num_workers = 16  # CPU 코어 수에 맞게 조절

"""
metadata.json을 읽어 학습/검증 데이터로더를 생성
"""
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# 데이터셋 인스턴스 생성
train_dataset = SurfaceSegmentationDataset(metadata['train_data'], target_size, is_train=True)
valid_dataset = SurfaceSegmentationDataset(metadata['valid_data'], target_size, is_train=False)


filename = "id2label.json"

# 클래스 이름과 인덱스 매핑 (배경은 0)
class_to_idx = {
    'background': 0,
    'caution_zone': 1,
    'bike_lane': 2,
    'alley': 3,
    'roadway': 4,
    'braille_guide_blocks': 5,
    'sidewalk': 6
}

# id2label = ????
# id2label = {int(k): v for k, v in id2label.items()}
# label2id = {v: k for k, v in id2label.items()}
# num_labels = len(id2label)

# class_to_idx로부터 id2label과 label2id를 생성합니다.
id2label = {int(idx): label for label, idx in class_to_idx.items()}
label2id = class_to_idx
num_labels = len(id2label)

print(f"클래스 수: {num_labels}")
print(f"클래스 레이블: {id2label}")


# =============================================================================
# 2. 이미지 프로세서 및 데이터 변환 설정
# =============================================================================

processor = SegformerImageProcessor()

# data augmentation!!
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

print("데이터 변환 설정 완료")

# =============================================================================
# 3. 모델 로드
# =============================================================================

pretrained_model_name = "nvidia/mit-b0"
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    id2label=id2label,
    label2id=label2id
)

print(f"모델 로드 완료: {pretrained_model_name}")
print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# =============================================================================
# 5. 훈련 설정
# =============================================================================

# 하이퍼파라미터 설정
epochs = 100
lr = 0.00006
batch_size = 64


# 데이터로더 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

import torch.nn.functional as F

class DirectSegFormer(nn.Module):
    """
    직접적으로 텐서를 처리하는 SegFormer
    """
    
    def __init__(self, pretrained_model_name="nvidia/mit-b0", num_classes=7):
        super().__init__()
        
        # 원본 모델 로드
        self.original_model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # 정규화 파라미터
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 이미지 텐서 (0-1 범위)
        """
        # 정규화
        x = (x - self.mean) / self.std
        
        # 원본 모델의 forward 방식 사용
        # pixel_values를 직접 전달
        outputs = self.original_model(pixel_values=x)
        
        return outputs.logits

    def predict(self, x):
        """예측 함수"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = F.softmax(logits, dim=1)
            pred_masks = torch.argmax(predictions, dim=1)
        return predictions, pred_masks


device= torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = DirectSegFormer(num_classes=7)
model.to(device)

import torch.optim as optim
# 옵티마이저와 손실 함수 정의
optimizer = optim.AdamW(model.parameters(), lr=lr)
# 다중 클래스 분할을 위한 표준 손실 함수
# 레이블 마스크는 (B, H, W) 형태이고, 각 픽셀 값은 클래스 인덱스여야 함
criterion = nn.CrossEntropyLoss()

import wandb
# wandb 초기화
wandb.init(project="segmentation_project", name=f"run_0707")

model.train()
for epoch in range(epochs):
    running_loss = 0
    for i, data in enumerate(train_loader):
        images = data['pixel_values'].to(device)
        masks= data['labels'].to(device)

        # 모델 순전파 -> 작은 해상도의 logits 출력
        # logits.shape: [B, num_classes, H/4, W/4]
        logits = model(images)

        # 🚀 Logits 업샘플링 (가장 중요한 부분) 🚀
        # upsampled_logits.shape: [B, num_classes, H, W]
        upsampled_logits = F.interpolate(
            logits,
            size=masks.shape[-2:],  # 원본 마스크의 (H, W) 크기로
            mode='bilinear',
            align_corners=False
        )

        optimizer.zero_grad()

        # CrossEntropyLoss는 (B, C, H, W) 형태의 logits와 (B, H, W) 형태의 마스크를 입력으로 받음
        loss = criterion(upsampled_logits, masks)
        loss.backward()
        optimizer.step()
        

        running_loss += loss.item()
        # if (i + 1) % 10 == 0:  # 10 배치마다 로그 출력
        print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        # break

    wandb.log({"loss": loss.item(), "epoch": epoch + 1, "step": i + 1})

    # 모델 저장
    model_save_path = f"ckpts/seg_model_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"모델이 저장되었습니다: {model_save_path}")


    # 간단한 추론 (학습 후 평가 또는 시각화용)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_data in valid_loader:  # val_loader는 미리 정의되어 있어야 함
            val_imgs = val_data['pixel_values'].to(device)
            val_masks = val_data['labels'].to(device)

            val_logits = model(val_imgs)

            val_outputs = F.interpolate(
                val_logits,
                size=masks.shape[-2:],  # 원본 마스크의 (H, W) 크기로
                mode='bilinear',
                align_corners=False
            )

            # loss 계산
            loss = criterion(val_outputs, val_masks)
            val_loss += loss.item()
            # wandb에 이미지 로그 (첫 배치만 예시로 기록)
            # wandb.log({
            #     "val_input": [wandb.Image(val_imgs[0].cpu())],
            #     "val_pred": [wandb.Image(preds[0].cpu())],
            #     "val_mask": [wandb.Image(val_masks[0].cpu())]
            # })
            # break  # 첫 배치만 시각화    

    # 평균 검증 손실을 wandb에 기록
    avg_val_loss = val_loss / len(valid_loader)
    # wandb.log({"val_loss": avg_val_loss, "epoch": epoch + 1})
print("학습 완료!")

wandb.finish()

    